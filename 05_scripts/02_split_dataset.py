"""02_split_dataset.py
Сплит датасета на train/val/test БЕЗ утечки данных.
Кадры группируются по видео-сегменту (task + video + file_index).
Все кадры из одного сегмента идут в один сплит.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_split(key: str, train_ratio: float, val_ratio: float) -> str:
    """Детерминированное распределение по сплиту на основе хэша ключа."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(h, 16) % 100
    train_cut = int(train_ratio * 100)
    val_cut = train_cut + int(val_ratio * 100)
    if bucket < train_cut:
        return "train"
    elif bucket < val_cut:
        return "val"
    return "test"


def extract_group_key(unique_name: str) -> str:
    """Извлекает ключ группы из имени файла.
    Формат: task_N__<video_name>frameXXX_fileY_(picket).jpg
    Ключ: task_N__<video_name>_fileY -- все кадры одного видео-сегмента
    попадут в один сплит, исключая утечку данных."""
    # Убираем расширение
    stem = Path(unique_name).stem
    # Ищем паттерн: всё до frameXXX + fileY
    m = re.match(r"^(.+?)frame\d+_(file\d+)", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
        
    # Ищем паттерн для новых данных: picket_XXX
    m2 = re.search(r"picket_(\d+)", stem)
    if m2:
        # Группируем пикеты блоками по 100 (чтобы соседние кадры попадали в один сплит)
        # Это предотвращает утечку данных (data leakage) между train и val
        block_idx = int(m2.group(1)) // 100
        parts = stem.split("__", 1)
        task_name = parts[0] if len(parts) > 1 else "task_unknown"
        return f"{task_name}_picketblock_{block_idx}"

    # Fallback: хэшируем по task_name (первая часть до __)
    parts = stem.split("__", 1)
    return parts[0] if len(parts) > 1 else stem


def collect_pairs(raw_data_dir: Path, image_extensions: list[str]) -> list[tuple[Path, Path, str]]:
    """Возвращает список (image_path, label_path, unique_name) по всем task_* папкам."""
    pairs = []
    task_dirs = sorted([p for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("task_")])

    for task_dir in task_dirs:
        images_dir = task_dir / "images"
        labels_dir = task_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"[ПРОПУСК] Нет label-файла для {img_path.name} ({task_dir.name})")
                continue
            unique_name = f"{task_dir.name}__{img_path.name}"
            pairs.append((img_path, label_path, unique_name))

    return pairs


def build_dataset(pairs: list[tuple[Path, Path, str]], output_dir: Path,
                   train_ratio: float, val_ratio: float, mode: str) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}

    for split in counts:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Группируем пары по видео-сегменту
    groups = defaultdict(list)
    for img_path, label_path, unique_name in pairs:
        group_key = extract_group_key(unique_name)
        groups[group_key].append((img_path, label_path, unique_name))

    # Сплит по ГРУППАМ -- все кадры одного видео идут в один сплит
    group_splits = {}
    for group_key in groups:
        group_splits[group_key] = get_split(group_key, train_ratio, val_ratio)

    print(f"Групп (видео-сегментов): {len(groups)}")
    for split_name in ["train", "val", "test"]:
        n = sum(1 for s in group_splits.values() if s == split_name)
        print(f"  {split_name}: {n} групп")

    for group_key, group_pairs in groups.items():
        split = group_splits[group_key]
        for img_path, label_path, unique_name in group_pairs:
            counts[split] += 1
            label_unique_name = Path(unique_name).stem + ".txt"
            dst_img = output_dir / "images" / split / unique_name
            dst_label = output_dir / "labels" / split / label_unique_name

            if mode == "symlink":
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                if not dst_label.exists():
                    dst_label.symlink_to(label_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(label_path, dst_label)

    return counts


def write_data_yaml(output_dir: Path, class_names: dict[int, str]):
    # ВАЖНО: class_names ключует по НАСТОЯЩЕМУ coco_id (1, 2, 3...), как в instances_task*.json.
    # А класс в .txt-файлах (то, что реально видит YOLO) — это coco_id - 1 (0-based).
    # Это подтверждено эмпирически скриптом verify_class_mapping.py на всех 5 тасках
    # (Гипотеза A: txt_class_id = coco_id - 1 — совпадает всегда).
    # Поэтому здесь явно делаем сдвиг при построении списка имён для data.yaml.
    max_coco_id = max(class_names.keys())
    max_txt_id = max_coco_id - 1

    names_list = [
        class_names.get(txt_id + 1, f"unknown_{txt_id}")
        for txt_id in range(max_txt_id + 1)
    ]

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names_list)},
    }

    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    script_dir = Path(__file__).resolve().parent
    root = (script_dir / cfg["project"]["root_dir"]).resolve()
    raw_data_dir = root / cfg["project"]["raw_data_dir"]
    output_dir = root / cfg["project"]["output_dataset_dir"] / "all_data"
    reports_dir = root / cfg["project"]["reports_dir"]
    image_extensions = cfg["dataset"]["image_extensions"]

    train_ratio = cfg["split"]["train"]
    val_ratio = cfg["split"]["val"]
    test_ratio = cfg["split"]["test"]
    mode = cfg["split"].get("mode", "copy")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        sys.exit("[ОШИБКА] Пропорции train/val/test в config.yaml должны суммироваться в 1.0")

    # Берём согласованную карту классов из отчёта EDA (не пересчитываем заново —
    # это гарантирует, что сплит использует ту же проверенную карту классов)
    eda_report_path = reports_dir / "eda_raw_report.json"
    if not eda_report_path.exists():
        sys.exit(f"[ОШИБКА] Не найден {eda_report_path}. Сначала запусти 01_eda_raw.py")

    with open(eda_report_path, "r", encoding="utf-8") as f:
        eda_report = json.load(f)
    merged_categories = {int(k): v for k, v in eda_report["merged_categories"].items()}

    print("Собираю пары (image, label) из всех task_* папок...")
    pairs = collect_pairs(raw_data_dir, image_extensions)
    print(f"Найдено {len(pairs)} валидных пар image+label")

    if output_dir.exists():
        print(f"[ПРЕДУПРЕЖДЕНИЕ] {output_dir} уже существует — файлы будут добавлены/перезаписаны")

    counts = build_dataset(pairs, output_dir, train_ratio, val_ratio, mode)
    print(f"\nРезультат сплита: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"Фактические пропорции: "
          f"{counts['train']/len(pairs):.1%} / {counts['val']/len(pairs):.1%} / {counts['test']/len(pairs):.1%}")

    write_data_yaml(output_dir, merged_categories)
    print(f"\n[OK] data.yaml создан в {output_dir / 'data.yaml'}")
    print(f"[OK] Датасет готов в {output_dir}")


if __name__ == "__main__":
    main()
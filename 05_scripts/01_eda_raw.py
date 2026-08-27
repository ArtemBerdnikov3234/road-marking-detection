"""
01_eda_raw.py

Шаг 1 подготовительного этапа.

Что делает:
  1. Обходит все task_* папки в 01_raw_data.
  2. Читает instances_task*.json (COCO) из каждой — сверяет category_id -> category_name
     между всеми тасками. Если хотя бы в одной паре тасков id одного и того же класса
     не совпадает — пайплайн ОСТАНАВЛИВАЕТСЯ (это критическая ошибка, дальше сплитовать нельзя).
  3. Считает статистику по images/ и labels/*.txt внутри каждого таска:
       - кол-во картинок, кол-во label-файлов, кол-во "пустых" (без разметки) картинок
       - распределение объектов по классам (по факту, из .txt, а не из COCO)
  4. Сохраняет отчёт в 06_reports/eda_raw_report.json и печатает summary в консоль.

Запуск:
    python 01_eda_raw.py --config config.yaml
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_task_dirs(raw_data_dir: Path) -> list[Path]:
    task_dirs = sorted([p for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("task_")])
    if not task_dirs:
        sys.exit(f"[ОШИБКА] Не найдено ни одной папки task_* в {raw_data_dir}")
    return task_dirs


def load_categories(task_dir: Path) -> dict[int, str]:
    """Находит instances_task*.json (COCO) или data.yaml (YOLO) и возвращает {category_id: name}."""
    json_candidates = list(task_dir.glob("instances_task*.json"))
    if not json_candidates:
        json_candidates = list(task_dir.glob("*.json"))
        
    if json_candidates:
        with open(json_candidates[0], "r", encoding="utf-8") as f:
            coco = json.load(f)
        categories = {c["id"]: str(c["name"]) for c in coco.get("categories", [])}
        if not categories:
            sys.exit(f"[ОШИБКА] В {json_candidates[0]} пустой или отсутствующий блок 'categories'")
        return categories
        
    yaml_candidates = list(task_dir.glob("data.yaml"))
    if yaml_candidates:
        with open(yaml_candidates[0], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # В YOLO data.yaml ID классов начинаются с 0 (как в .txt).
        # В COCO JSON они начинаются с 1. Приводим к 1-based для сверки:
        categories = {int(k) + 1: str(v) for k, v in cfg.get("names", {}).items()}
        return categories

    sys.exit(f"[ОШИБКА] В {task_dir} не найден ни instances_task*.json (COCO), ни data.yaml (YOLO)")


def check_category_consistency(task_categories: dict[str, dict[int, str]]) -> dict[int, str]:
    """
    Сверяет category_id -> name между всеми тасками.
    Возвращает единую (согласованную) карту классов, если всё ок.
    Если найдено расхождение — печатает подробный конфликт и прерывает выполнение.
    """
    reference_task = next(iter(task_categories))
    reference_map = task_categories[reference_task]

    conflicts = []
    for task_name, cat_map in task_categories.items():
        for cat_id, cat_name in cat_map.items():
            if cat_id in reference_map and reference_map[cat_id] != cat_name:
                conflicts.append(
                    f"  id={cat_id}: в '{reference_task}' -> '{reference_map[cat_id]}', "
                    f"в '{task_name}' -> '{cat_name}'"
                )

    if conflicts:
        print("\n[КРИТИЧЕСКАЯ ОШИБКА] Нумерация классов расходится между task_* папками!")
        print("Объединять эти данные в один датасет НЕЛЬЗЯ, пока это не исправлено:")
        print("\n".join(conflicts))
        print(
            "\nВарианты решения:\n"
            "  1. Переэкспортировать разметку из CVAT с единым списком категорий для всех тасков.\n"
            "  2. Написать скрипт remap классов по названию (name -> единый id) перед сплитом.\n"
        )
        sys.exit(1)

    # объединяем все карты в одну (расхождений нет, можно смёржить)
    merged = {}
    for cat_map in task_categories.values():
        merged.update(cat_map)
    return merged


def analyze_task(task_dir: Path, image_extensions: list[str]) -> dict:
    images_dir = task_dir / "images"
    labels_dir = task_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ПРЕДУПРЕЖДЕНИЕ] В {task_dir} нет images/ или labels/ — пропускаю статистику по файлам")
        return {"images": 0, "labels": 0, "empty_labels": 0, "class_counts": {}}

    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}

    missing_labels = [p.name for p in image_files if p.stem not in label_files]
    empty_labels = 0
    class_counter = Counter()

    for stem, label_path in label_files.items():
        lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            class_id = int(line.split()[0])
            class_counter[class_id] += 1

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # root_dir считается от расположения ЭТОГО файла, а не от cwd терминала —
    # так скрипт работает одинаково, откуда бы его ни запустили.
    script_dir = Path(__file__).resolve().parent
    root = (script_dir / cfg["project"]["root_dir"]).resolve()
    raw_data_dir = root / cfg["project"]["raw_data_dir"]
    reports_dir = root / cfg["project"]["reports_dir"]

    if not raw_data_dir.exists():
        sys.exit(
            f"[ОШИБКА] Не найдена папка с исходными данными: {raw_data_dir}\n"
            f"Проверь путь raw_data_dir в config.yaml."
        )
    reports_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = cfg["dataset"]["image_extensions"]

    task_dirs = find_task_dirs(raw_data_dir)
    print(f"Найдено {len(task_dirs)} task_* папок: {[t.name for t in task_dirs]}\n")

    # 1. Проверка согласованности категорий
    task_categories = {t.name: load_categories(t) for t in task_dirs}
    merged_categories = check_category_consistency(task_categories)
    print(f"[OK] Категории согласованы между всеми тасками. Всего классов: {len(merged_categories)}")

    if len(merged_categories) != cfg["dataset"]["num_classes"]:
        print(
            f"[ПРЕДУПРЕЖДЕНИЕ] В конфиге указано {cfg['dataset']['num_classes']} классов (ГОСТ), "
            f"а в разметке найдено {len(merged_categories)}. Проверь — возможно, не все классы "
            f"встретились в текущих данных, это не всегда ошибка."
        )

    # 2. Статистика по каждому таску
    per_task_stats = {}
    total_class_counts = Counter()
    total_images, total_labels, total_empty = 0, 0, 0

    for task_dir in task_dirs:
        stats = analyze_task(task_dir, image_extensions)
        per_task_stats[task_dir.name] = stats
        total_images += stats["images"]
        total_labels += stats["labels"]
        total_empty += stats["empty_labels"]
        total_class_counts.update(stats["class_counts"])
        if stats.get("missing_labels"):
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] {task_dir.name}: {len(stats['missing_labels'])} "
                f"изображений без соответствующего label-файла"
            )

    # 3. Печать сводки по дисбалансу классов
    # ВАЖНО: total_class_counts собран из .txt-файлов, там class_id 0-based.
    # merged_categories собран из COCO JSON, там id 1-based (id самой категории).
    # Подтверждено verify_class_mapping.py: txt_class_id = coco_id - 1 для всех тасков.
    # Поэтому здесь явно сдвигаем при сопоставлении, иначе название класса съезжает на одну позицию.
    print("\n--- Распределение объектов по классам (по всем тасками) ---")
    max_count = max(total_class_counts.values()) if total_class_counts else 0
    for coco_id in sorted(merged_categories):
        name = merged_categories[coco_id]
        txt_class_id = coco_id - 1
        count = total_class_counts.get(txt_class_id, 0)
        bar = "#" * int(30 * count / max_count) if max_count else ""
        flag = "  <-- РЕДКИЙ КЛАСС" if max_count and count < 0.05 * max_count else ""
        print(f"  [coco_id={coco_id:>2} / txt_id={txt_class_id:>2}] {name:<25} {count:>6} {bar}{flag}")

    print(f"\nВсего изображений: {total_images}, всего label-файлов: {total_labels}, "
          f"из них пустых (без объектов): {total_empty}")

    # 4. Сохранение отчёта
    report = {
        "merged_categories": merged_categories,
        "per_task_stats": per_task_stats,
        "total_class_counts": dict(total_class_counts),
        "total_images": total_images,
        "total_labels": total_labels,
        "total_empty_labels": total_empty,
    }
    report_path = reports_dir / "eda_raw_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Отчёт сохранён в {report_path}")


if __name__ == "__main__":
    main()
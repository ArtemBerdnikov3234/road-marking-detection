"""
03_validate_dataset.py

Шаг 3 (финальный) подготовительного этапа. Запускать после 02_split_dataset.py
и ПЕРЕД тем, как отдавать data.yaml в обучение YOLO.

Проверяет:
  1. Для каждого images/{split}/*.jpg есть labels/{split}/*.txt и наоборот (нет "сирот").
  2. Все class_id в .txt-файлах находятся в диапазоне [0, num_classes-1].
  3. Каждая строка label-файла имеет корректное число координат (нечётное
     кол-во значений после class_id недопустимо для polygon-формата).
  4. Нет файлов с одинаковым именем, попавших одновременно в разные сплиты
     (утечка данных).
  5. Печатает финальную сводку — если всё ок, датасет готов к обучению.

Запуск:
    python 03_validate_dataset.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_split(images_dir: Path, labels_dir: Path, num_classes: int) -> dict:
    image_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    bad_class_ids = []
    malformed_lines = []

    for label_file in labels_dir.glob("*.txt"):
        for line_num, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = parts[1:]

            if class_id < 0 or class_id >= num_classes:
                bad_class_ids.append(f"{label_file.name}:{line_num} class_id={class_id}")

            # Формат может быть:
            # - Detection (4 координаты: x_center, y_center, width, height)
            # - Segmentation (polygon: >=6 координат, чётное число x1,y1,x2,y2...)
            if len(coords) == 4:
                pass # Bounding box — OK
            elif len(coords) % 2 != 0 or len(coords) < 6:
                malformed_lines.append(f"{label_file.name}:{line_num} ({len(coords)} координат)")

    return {
        "images_without_labels": sorted(images_without_labels),
        "labels_without_images": sorted(labels_without_images),
        "bad_class_ids": bad_class_ids,
        "malformed_lines": malformed_lines,
        "num_images": len(image_stems),
        "num_labels": len(label_stems),
    }


def check_cross_split_leakage(output_dir: Path, splits: list[str]) -> list[str]:
    """Проверяет, что один и тот же файл (по имени) не попал в два сплита одновременно."""
    seen = {}
    leaks = []
    for split in splits:
        images_dir = output_dir / "images" / split
        if not images_dir.exists():
            continue
        for p in images_dir.iterdir():
            if p.name in seen and seen[p.name] != split:
                leaks.append(f"{p.name} найден и в '{seen[p.name]}', и в '{split}'")
            seen[p.name] = split
    return leaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    script_dir = Path(__file__).resolve().parent
    root = (script_dir / cfg["project"]["root_dir"]).resolve()
    output_dir = root / cfg["project"]["output_dataset_dir"] / "all_data"
    num_classes = cfg["dataset"]["num_classes"]
    splits = ["train", "val", "test"]

    if not (output_dir / "data.yaml").exists():
        sys.exit(f"[ОШИБКА] Не найден {output_dir / 'data.yaml'}. Сначала запусти 02_split_dataset.py")

    all_ok = True
    total_images = 0

    for split in splits:
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        if not images_dir.exists():
            print(f"[ОШИБКА] Отсутствует папка {images_dir}")
            all_ok = False
            continue

        result = validate_split(images_dir, labels_dir, num_classes)
        total_images += result["num_images"]

        print(f"\n--- {split} ---")
        print(f"  Изображений: {result['num_images']}, label-файлов: {result['num_labels']}")

        if result["images_without_labels"]:
            all_ok = False
            print(f"  [ОШИБКА] {len(result['images_without_labels'])} изображений без label-файла: "
                  f"{result['images_without_labels'][:5]}{'...' if len(result['images_without_labels']) > 5 else ''}")
        if result["labels_without_images"]:
            all_ok = False
            print(f"  [ОШИБКА] {len(result['labels_without_images'])} label-файлов без изображения")
        if result["bad_class_ids"]:
            all_ok = False
            print(f"  [ОШИБКА] Некорректные class_id (вне диапазона 0-{num_classes-1}): "
                  f"{result['bad_class_ids'][:5]}")
        if result["malformed_lines"]:
            all_ok = False
            print(f"  [ОШИБКА] Некорректные строки разметки (не 4, и не чётное >=6): "
                  f"{result['malformed_lines'][:5]}")

        if not any([result["images_without_labels"], result["labels_without_images"],
                    result["bad_class_ids"], result["malformed_lines"]]):
            print("  [OK] Сплит корректен")

    leaks = check_cross_split_leakage(output_dir, splits)
    if leaks:
        all_ok = False
        print(f"\n[ОШИБКА] Обнаружена утечка данных между сплитами:")
        for leak in leaks[:10]:
            print(f"  {leak}")

    print(f"\nВсего изображений в датасете: {total_images}")


if __name__ == "__main__":
    main()

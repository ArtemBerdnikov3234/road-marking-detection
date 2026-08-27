"""
04_filter_horizontal_only.py
Фильтрует датасет: оставляет только горизонтальную разметку, убирает продольные линии (1.1–1.11).
"""

import shutil
import sys
from pathlib import Path

import yaml


# === Конфигурация ===

# Классы для УДАЛЕНИЯ (продольная разметка 1.1–1.11, txt_id 0–10)
EXCLUDE_IDS = set(range(0, 11))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# Маппинг старых id → новых id (автоматический)
# Старый id=11 (1.12) → новый id=0, id=12 (1.13) → новый id=1, ...
OLD_TO_NEW = {}
new_id = 0
for old_id in range(37):
    if old_id not in EXCLUDE_IDS:
        OLD_TO_NEW[old_id] = new_id
        new_id += 1

# Имена классов из текущего data.yaml (только горизонтальные)
HORIZONTAL_NAMES = {
    0: "1.12",
    1: "1.13",
    2: "1.14.1",
    3: "1.14.2",
    4: "1.14.3",
    5: "1.15",
    6: "1.16.1",
    7: "1.16.2",
    8: "1.16.3",
    9: "1.17.1",
    10: "1.17.2",
    11: "1.18",
    12: "1.19",
    13: "1.20",
    14: "1.22",
    15: "1.23.1",
    16: "1.23.2",
    17: "1.23.3",
    18: "1.24.1",
    19: "1.24.2",
    20: "1.24.3",
    21: "1.24.4",
    22: "1.25",
    23: "1.26",
    24: "ШП",
    25: "1.21",
}


# === Ручное перераспределение редких классов между сплитами ===
# Обычный хэш-сплит не учитывает классы, поэтому несколько классов с 6-22 объектами
# на весь датасет оказались без единого примера в val и/или test (проверено
# check_per_class_split_balance.py, кандидаты найдены find_rebalance_candidates.py).
# Эти файлы ПРИНУДИТЕЛЬНО кладутся в указанный сплит, независимо от исходного хэша.
# Проверено вручную: ни один перенос не создаёт новую дыру для других классов
# (соседние классы в этих же файлах — 1.24.2, 1.16.1, 1.16.2, 1.14.1 — везде с запасом).
SPLIT_OVERRIDES = {
    "task_4__frame300_file1_(000+343).jpg": "val",                          # 1.23.1 -> val
    "task_3__frameShotsframe820_file0_(-000+838-).jpg": "val",               # 1.23.2 + 1.23.3 -> val
    "task_3__ViewRoadSFF2frame0_file1_(-000+002-).jpg": "test",              # 1.23.2 + 1.23.3 -> test
    "task_3__task_3frame1996_file1_(-003+143-).jpg": "test",                # 1.19 -> test
    "task_1__изображения_дорогframe18_file0_(000+013).jpg": "test",         # 1.20 -> test
    "task_2__road_markingsframe34_file2_(000+036).jpg": "test",             # 1.24.3 -> test
    "task_1__изображения_дорогframe67_file0_(000+068).jpg": "test",         # 1.24.4 -> test
    
    # Новые добавления для исправления нулей в val/test (Классы: 13, 20, 21, 22, 23, 24)
    "task_6__picket_643.jpg": "val",                                         # Class 13 (1.20) -> val
    "task_3__frameShotsframe239_file0_(-000+254-).jpg": "val",               # Class 20 (1.24.3) -> val
    "task_6__picket_1933.jpg": "val",                                        # Class 21 (1.24.4) -> val
    "task_6__picket_1945.jpg": "val",                                        # Class 21 (1.24.4) -> val
    "task_2__road_markingsframe34_file1_(000+036).jpg": "val",               # Class 22 (1.25) -> val
    "task_2__road_markingsframe95_file1_(000+099).jpg": "val",               # Class 22 (1.25) -> val
    "task_4__frameShotsframe1001_file2_(001+125).jpg": "test",               # Class 23 (1.26) -> test
    "task_4__frameShotsframe495_file0_(000+541).jpg": "test",                # Class 23 (1.26) -> test
    "task_5__frameShotsframe7901_file0_(024+912).jpg": "train",              # Class 24 (ШП) -> train
    "task_5__frameShotsframe7371_file1_(024+032).jpg": "train",              # Class 24 (ШП) -> train
    "task_5__frameShotsframe3952_file1_(018+264).jpg": "train",              # Class 24 (ШП) -> train
    "task_5__frameShotsframe6768_file1_(029+260).jpg": "val",                # Class 24 (ШП) -> val
    "task_5__frameShotsframe7368_file1_(024+027).jpg": "val",                # Class 24 (ШП) -> val
}


def filter_label_file(src_label: Path) -> list[str]:
    """Читает label-файл, удаляет продольные классы, перенумеровывает остальные.

    Returns:
        Список отфильтрованных строк (может быть пустым).
    """
    lines = src_label.read_text(encoding="utf-8").strip().splitlines()
    filtered = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        old_class_id = int(parts[0])

        if old_class_id in EXCLUDE_IDS:
            continue  # пропускаем продольную разметку

        if old_class_id not in OLD_TO_NEW:
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Неизвестный class_id={old_class_id} в {src_label.name}, пропускаю")
            continue

        new_class_id = OLD_TO_NEW[old_class_id]
        # Заменяем class_id, остальные координаты оставляем как есть
        new_line = str(new_class_id) + " " + " ".join(parts[1:])
        filtered.append(new_line)

    return filtered


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    src_dataset = project_root / "02_yolo_dataset" / "all_data"
    dst_dataset = project_root / "02_yolo_dataset" / "horizontal"

    if not src_dataset.exists():
        sys.exit(f"[ОШИБКА] Не найден исходный датасет: {src_dataset}")

    if dst_dataset.exists():
        shutil.rmtree(dst_dataset)

    print(f"Исходный датасет: {src_dataset}")
    print(f"Целевой датасет:  {dst_dataset}")
    print(f"Удаляемые классы (продольные 1.1–1.11): txt_id {sorted(EXCLUDE_IDS)}")
    print(f"Оставляемые классы (горизонтальные): {len(HORIZONTAL_NAMES)}")
    print(f"Маппинг id: {OLD_TO_NEW}")
    print()

    stats = {"total_images": 0, "kept_images": 0, "skipped_empty": 0,
             "total_objects": 0, "kept_objects": 0, "removed_objects": 0}

    # Заранее создаём все выходные папки — файлы могут переезжать в ДРУГОЙ сплит через SPLIT_OVERRIDES
    for split in ["train", "val", "test"]:
        (dst_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_kept_counts = {"train": 0, "val": 0, "test": 0}
    split_skipped_counts = {"train": 0, "val": 0, "test": 0}
    override_applied = []

    for split in ["train", "val", "test"]:
        src_images = src_dataset / "images" / split
        src_labels = src_dataset / "labels" / split

        if not src_images.exists():
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Нет папки {src_images}, пропускаю")
            continue

        for img_path in sorted(src_images.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            stats["total_images"] += 1
            label_path = src_labels / f"{img_path.stem}.txt"

            # Куда файл реально пойдёт — обычно тот же сплит, но SPLIT_OVERRIDES может переопределить
            effective_split = SPLIT_OVERRIDES.get(img_path.name, split)
            if effective_split != split:
                override_applied.append((img_path.name, split, effective_split))

            dst_images = dst_dataset / "images" / effective_split
            dst_labels = dst_dataset / "labels" / effective_split

            if not label_path.exists():
                stats["skipped_empty"] += 1
                split_skipped_counts[effective_split] += 1
                continue

            # Считаем объекты до фильтрации
            original_lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            stats["total_objects"] += len(original_lines)

            # Фильтруем
            filtered_lines = filter_label_file(label_path)
            stats["removed_objects"] += len(original_lines) - len(filtered_lines)
            stats["kept_objects"] += len(filtered_lines)

            # Пропускаем изображения, где не осталось горизонтальной разметки
            if not filtered_lines:
                stats["skipped_empty"] += 1
                split_skipped_counts[effective_split] += 1
                continue

            # Копируем изображение + сохраняем отфильтрованный label — в EFFECTIVE сплит
            shutil.copy2(img_path, dst_images / img_path.name)
            (dst_labels / f"{img_path.stem}.txt").write_text(
                "\n".join(filtered_lines) + "\n", encoding="utf-8"
            )
            stats["kept_images"] += 1
            split_kept_counts[effective_split] += 1

    if override_applied:
        print(f"Применено принудительных переносов между сплитами: {len(override_applied)}")
        for name, from_split, to_split in override_applied:
            print(f"    {name}:  {from_split} -> {to_split}")
        print()

    for split in ["train", "val", "test"]:
        print(f"  {split}: сохранено {split_kept_counts[split]}, пропущено {split_skipped_counts[split]} "
              f"(без горизонтальной разметки)")

    # Генерация data.yaml
    data_yaml = {
        "path": str(dst_dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": HORIZONTAL_NAMES,
    }
    yaml_path = dst_dataset / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n{'=' * 50}")
    print(f"ИТОГО:")
    print(f"  Изображений всего:          {stats['total_images']}")
    print(f"  Изображений сохранено:      {stats['kept_images']}")
    print(f"  Изображений пропущено:      {stats['skipped_empty']} (нет горизонтальной разметки)")
    print(f"  Объектов всего:             {stats['total_objects']}")
    print(f"  Объектов оставлено:         {stats['kept_objects']} (горизонтальная разметка)")
    print(f"  Объектов удалено:           {stats['removed_objects']} (продольные 1.1–1.11)")
    print(f"  Классов:                    {len(HORIZONTAL_NAMES)}")
    print(f"[OK] data.yaml: {yaml_path}")
    print(f"[OK] Датасет {dst_dataset.name} успешно создан.")


if __name__ == "__main__":
    main()
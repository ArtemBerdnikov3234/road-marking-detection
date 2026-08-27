"""
05_oversample_rare_classes.py
Random Oversampling редких классов + цветовая аугментация — ТОЛЬКО для train.
val и test не трогаются.
"""

import argparse
from collections import Counter
from pathlib import Path

# pyrefly: ignore [missing-import]
import albumentations as A
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import yaml

AUG_PREFIX = "aug"

# 8 разных цветовых аугментаций для максимального разнообразия.
# Геометрия НЕ меняется — метки (bbox/polygon) остаются валидными.
AUGMENTATIONS = [
    # 1. Затемнение
    A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.15), contrast_limit=0.05, p=1.0),
    # 2. Размытие
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    # 3. Цветовой шум
    A.GaussNoise(std_range=(0.06, 0.16), p=1.0),
    # 4. Контраст + баланс белого
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(0.2, 0.4), p=1.0),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    ]),
    # 5. Осветление (солнце)
    A.RandomBrightnessContrast(brightness_limit=(0.15, 0.35), contrast_limit=(-0.1, 0.1), p=1.0),
    # 6. Сильное размытие + затемнение (дождь/туман)
    A.Compose([
        A.GaussianBlur(blur_limit=(5, 11), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.05), contrast_limit=(-0.15, 0.0), p=1.0),
    ]),
    # 7. Тональный сдвиг (другое время суток)
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
    # 8. CLAHE + шум (камера низкого качества)
    A.Compose([
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        A.GaussNoise(std_range=(0.03, 0.08), p=1.0),
    ]),
]


def clean_previous_augmentations(images_dir: Path, labels_dir: Path):
    removed = 0
    for p in list(images_dir.glob(f"{AUG_PREFIX}*__*")) + list(labels_dir.glob(f"{AUG_PREFIX}*__*")):
        p.unlink()
        removed += 1
    if removed:
        print(f"[INFO] Удалено {removed} файлов от прошлого запуска oversampling (идемпотентность)")


def load_image_class_counts(labels_dir: Path) -> dict[str, Counter]:
    """Для каждой картинки (по stem) — Counter{class_id: сколько ОБЪЕКТОВ этого класса в файле}."""
    result = {}
    for label_path in labels_dir.glob("*.txt"):
        if label_path.stem.startswith(AUG_PREFIX):
            continue
        counter = Counter()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counter[int(line.split()[0])] += 1
        if counter:
            result[label_path.stem] = counter
    return result


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    """Загружает имена классов из data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", {})


def main():
    parser = argparse.ArgumentParser(description="Oversampling редких классов в YOLO-датасете (только train)")
    parser.add_argument("--dataset", required=False, default="../02_yolo_dataset/horizontal", help="Путь к папке датасета")
    parser.add_argument("--target", type=int, default=80,
                         help="Целевое кол-во объектов класса в train (по умолчанию 80)")
    parser.add_argument("--max-dupes", type=int, default=5,
                         help="Максимум доп. копий одной картинки (по умолчанию 5)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Только показать план — не создавать файлы")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"

    if not train_labels.exists():
        print(f"[ОШИБКА] Не найдена папка {train_labels}")
        return

    class_names = load_class_names(dataset_dir)

    if not args.dry_run:
        clean_previous_augmentations(train_images, train_labels)

    image_class_counts = load_image_class_counts(train_labels)  # stem -> Counter
    running_counts = Counter()
    for counter in image_class_counts.values():
        running_counts.update(counter)

    original_counts = dict(running_counts)

    print(f"{'=' * 65}")
    print(f"  Oversampling редких классов")
    print(f"{'=' * 65}")
    print(f"  Датасет:     {dataset_dir.resolve()}")
    print(f"  Классов:     {len(running_counts)}")
    print(f"  Картинок:    {len(image_class_counts)}")
    print(f"  Target:      {args.target} объектов/класс")
    print(f"  Max dupes:   {args.max_dupes} копий/картинка")
    print(f"  Аугментаций: {len(AUGMENTATIONS)} вариантов")
    if args.dry_run:
        print(f"  Режим:       DRY RUN (файлы не создаются)")
    print()

    # --- Итеративный алгоритм подбора ---
    dupes_used = Counter()          # stem -> сколько доп. копий уже создано
    gave_up = set()                 # классы, для которых кандидаты закончились
    generation_order = []           # [(stem, copy_index)]

    while True:
        deficient = [c for c in running_counts
                     if running_counts[c] < args.target and c not in gave_up]
        if not deficient:
            break

        class_id = min(deficient, key=lambda c: running_counts[c])

        candidates = [
            stem for stem, counter in image_class_counts.items()
            if class_id in counter and dupes_used[stem] < args.max_dupes
        ]
        if not candidates:
            gave_up.add(class_id)
            continue

        def collateral_penalty(stem):
            # штраф — сколько объектов УЖЕ ЗАПОЛНЕННЫХ классов заодно утащит эта картинка
            counter = image_class_counts[stem]
            return sum(n for c, n in counter.items()
                       if c != class_id and running_counts[c] >= args.target)

        chosen = min(candidates, key=collateral_penalty)
        dupes_used[chosen] += 1
        generation_order.append((chosen, dupes_used[chosen]))
        running_counts.update(image_class_counts[chosen])

    if gave_up:
        gave_up_names = [class_names.get(c, str(c)) for c in sorted(gave_up)]
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось дотянуть до target (кончились картинки-кандидаты):")
        for c in sorted(gave_up):
            name = class_names.get(c, f"id_{c}")
            print(f"    id={c} ({name}): {running_counts[c]}/{args.target}")
        print()

    # --- Таблица «было → стало» ---
    print(f"{'id':>4}  {'Класс':<14} {'Было':>6}  {'Стало':>6}  {'Цель':>6}  {'Статус'}")
    print("-" * 62)
    needs_boost = 0
    already_ok = 0
    for class_id in sorted(original_counts):
        name = class_names.get(class_id, f"id_{class_id}")
        before = original_counts[class_id]
        after = running_counts[class_id]
        if class_id in gave_up:
            status = "[ПРЕДУПРЕЖДЕНИЕ] не дотянули"
            needs_boost += 1
        elif before < args.target and after >= args.target:
            status = "[OK] дополнен"
            needs_boost += 1
        elif before >= args.target:
            status = "— уже достаточно"
            already_ok += 1
        else:
            status = ""
            needs_boost += 1
        print(f"{class_id:>4}  {name:<14} {before:>6}  {after:>6}  {args.target:>6}  {status}")

    print(f"\nКлассов дополнено: {needs_boost}, уже достаточно: {already_ok}")
    print(f"Запланировано новых файлов: {len(generation_order)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Файлы не создавались. Уберите --dry-run для реального запуска.")
        return

    # --- Физически генерируем файлы ---
    total_new_images = 0
    total_gen = len(generation_order)
    for idx, (stem, copy_index) in enumerate(generation_order):
        img_candidates = [p for p in train_images.glob(f"{stem}.*")
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not img_candidates:
            print(f"  [ПРОПУСК] Нет картинки для {stem}")
            continue
        img_path = img_candidates[0]
        label_path = train_labels / f"{stem}.txt"
        label_content = label_path.read_text(encoding="utf-8")

        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        transform = AUGMENTATIONS[(copy_index - 1) % len(AUGMENTATIONS)]
        augmented = transform(image=image)["image"]

        new_stem = f"{AUG_PREFIX}{copy_index}__{stem}"
        new_img_path = train_images / f"{new_stem}{img_path.suffix}"
        new_label_path = train_labels / f"{new_stem}.txt"

        success, encoded = cv2.imencode(img_path.suffix, augmented)
        if success:
            encoded.tofile(str(new_img_path))
        new_label_path.write_text(label_content, encoding="utf-8")  # геометрия не меняется
        total_new_images += 1

        # Прогресс
        if (idx + 1) % 50 == 0 or (idx + 1) == total_gen:
            print(f"  [{idx + 1}/{total_gen}] файлов создано...")

    # --- Итоговая сводка ---
    print(f"\n{'=' * 65}")
    print(f"  [OK] ГОТОВО")
    print(f"{'=' * 65}")
    print(f"  Создано новых файлов: {total_new_images}")
    print(f"  Train теперь: {len(image_class_counts) + total_new_images} картинок "
          f"({len(image_class_counts)} оригиналов + {total_new_images} аугментированных)")
    print(f"  val и test НЕ тронуты — оценка модели остаётся честной.")
    
if __name__ == "__main__":
    main()
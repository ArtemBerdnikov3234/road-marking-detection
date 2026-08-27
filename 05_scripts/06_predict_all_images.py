"""
05_predict_all_images.py

Прогоняет обученную модель по ВСЕМ изображениям датасета (train + val + test)
и сохраняет визуализацию предсказаний в 06_reports/predictions_all/.

Запуск:
    python 05_predict_all_images.py
"""

import matplotlib
matplotlib.use("Agg")

import shutil
from pathlib import Path
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# === Настройки ===
PROJECT_ROOT = Path(r"E:\project\SibDor")
DATASET_DIR = PROJECT_ROOT / "02_yolo_dataset_horizontal"
MODEL_PATH = PROJECT_ROOT / "03_models" / "yolov8n_seg_horizontal_v2" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "06_reports" / "predictions_all"

CONF_THRESHOLD = 0.25       # порог уверенности (ниже — больше детекций, но больше шума)
IMGSZ = 1280                # разрешение inference (такое же, как при обучении)


def main():
    if not MODEL_PATH.exists():
        print(f"[ОШИБКА] Модель не найдена: {MODEL_PATH}")
        return

    # Очистка выходной папки
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    model = YOLO(str(MODEL_PATH))
    print(f"Модель: {MODEL_PATH.name}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"Разрешение inference: {IMGSZ}")
    print()

    total = 0
    for split in ["train", "val"]:
        images_dir = DATASET_DIR / "images" / split
        if not images_dir.exists():
            continue

        split_output = OUTPUT_DIR / split
        split_output.mkdir(parents=True, exist_ok=True)

        img_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        print(f"--- {split}: {len(img_files)} изображений ---")

        for i, img_path in enumerate(img_files):
            results = model.predict(
                str(img_path),
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                device="cpu",
                verbose=False,
            )

            # Сохраняем аннотированное изображение
            annotated = results[0].plot(
                line_width=2,
                font_size=10,
            )

            # pyrefly: ignore [missing-import]
            import cv2
            out_path = split_output / img_path.name
            cv2.imwrite(str(out_path), annotated)
            total += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(img_files):
                print(f"  [{i+1}/{len(img_files)}] сохранено")

    print(f"\n[OK] Все {total} предсказаний сохранены в {OUTPUT_DIR}")
    print(f"  train: {OUTPUT_DIR / 'train'}")
    print(f"  val:   {OUTPUT_DIR / 'val'}")
    print(f"  test:  {OUTPUT_DIR / 'test'}")


if __name__ == "__main__":
    main()

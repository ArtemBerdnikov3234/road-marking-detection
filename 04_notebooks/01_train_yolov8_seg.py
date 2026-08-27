# %% [markdown]
# # 🚀 Обучение YOLOv8n-seg — Дорожная разметка ГОСТ Р 51256-2018
# 
# **Датасет:** 248 изображений (170 train / 45 val / 33 test), 37 классов
# **GPU:** AMD Radeon RX 6600M через DirectML
# **Модель:** YOLOv8n-seg (nano segmentation, pretrained на COCO)

# %% [markdown]
# ## 1. Проверка окружения

# %%
import matplotlib
matplotlib.use("Agg")  # Не открывать окна — только сохранять в файл

import sys
import torch
# pyrefly: ignore [missing-import]
import torch_directml
from pathlib import Path

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

dml = torch_directml.device()
print(f"DirectML device: {dml}")
print(f"GPU: {torch_directml.device_name(0)}")

# Быстрый smoke-test: матричное умножение на GPU
t = torch.randn(256, 256).to(dml)
result = t @ t
print(f"GPU matmul test: OK (shape={result.shape})")
del t, result

# %% [markdown]
# ## 2. Проверка датасета

# %%
import yaml

# Пути
PROJECT_ROOT = Path(r"E:\project\SibDor")
# Переключено на отфильтрованный датасет (только горизонтальная разметка 1.12–1.26)
DATASET_DIR = PROJECT_ROOT / "02_yolo_dataset_horizontal"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "03_models"
MODELS_DIR.mkdir(exist_ok=True)

# Читаем data.yaml
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

print(f"Dataset path: {data_cfg['path']}")
print(f"Классов: {len(data_cfg['names'])}")
print(f"Классы: {data_cfg['names']}")

# Считаем файлы в каждом сплите
for split in ["train", "val", "test"]:
    imgs = list((DATASET_DIR / "images" / split).glob("*.*"))
    lbls = list((DATASET_DIR / "labels" / split).glob("*.txt"))
    status = "✅" if len(imgs) == len(lbls) else "❌ MISMATCH"
    print(f"  {split}: {len(imgs)} images, {len(lbls)} labels {status}")

# %% [markdown]
# ## 3. Визуальная проверка — несколько изображений с разметкой

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def draw_yolo_polygons(img_path, label_path, class_names, max_polygons=50):
    """Отрисовка полигонов YOLO-seg на изображении."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    if not label_path.exists():
        return img
    
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    overlay = img.copy()
    
    # Генерируем цвета для классов
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 255, size=(len(class_names), 3)).tolist()
    
    for line in lines[:max_polygons]:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + min 3 точки (6 координат)
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        # Денормализация координат
        points = []
        for i in range(0, len(coords) - 1, 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            points.append([px, py])
        
        if len(points) < 3:
            continue
        
        pts = np.array(points, dtype=np.int32)
        color = colors[cls_id % len(colors)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        
        # Подпись класса
        cx, cy = pts.mean(axis=0).astype(int)
        label = class_names.get(cls_id, str(cls_id))
        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Полупрозрачное наложение полигонов
    result = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    return result

# Выбираем 6 случайных изображений из train
train_imgs = sorted((DATASET_DIR / "images" / "train").glob("*.*"))
sample_imgs = random.sample(train_imgs, min(6, len(train_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, img_path in zip(axes.flat, sample_imgs):
    label_path = DATASET_DIR / "labels" / "train" / f"{img_path.stem}.txt"
    vis = draw_yolo_polygons(img_path, label_path, data_cfg["names"])
    if vis is not None:
        ax.imshow(vis)
    ax.set_title(img_path.name, fontsize=8)
    ax.axis("off")

plt.suptitle("Примеры train-изображений с разметкой", fontsize=14)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "06_reports" / "train_samples_preview.png"), dpi=150)
plt.show()
print("Превью сохранено в 06_reports/train_samples_preview.png")

# %% [markdown]
# ## 4. Обучение YOLOv8n-seg
# 
# **Важно про DirectML:** Ultralytics автоматически определяет `device`.
# Если DirectML вызывает проблемы — переключись на `device="cpu"` (будет медленнее, но гарантированно работает).

# %%
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# Загрузка pretrained модели (nano-сегментация)
# Модель скачана один раз и лежит в 03_models/
pretrained_path = PROJECT_ROOT / "03_models" / "yolov8n-seg.pt"
model = YOLO(str(pretrained_path))

print("Модель загружена: yolov8n-seg")
print(f"Параметров: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

# %%
# === ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ===
# Можно менять перед запуском — все в одном месте

TRAIN_CONFIG = dict(
    data=str(DATA_YAML),
    
    # --- Основные ---
    epochs=100,             # 20 было слишком мало — loss не успел сойтись
    imgsz=1280,             # 1280 вместо 640: стрелки и мелкая разметка лучше видны
    batch=4,                # уменьшили из-за imgsz=1280 (больше RAM на картинку)
    patience=30,            # early stopping: стоп если val loss не улучшается 30 эпох
    workers=4,              # загрузка данных; на Windows иногда нужно 0
    
    # --- Устройство ---
    # Ultralytics не поддерживает DirectML напрямую (только CUDA/CPU).
    # CPU на 170 train-изображениях — ~1-3 часа, это нормально для nano-модели.
    device="cpu",
    
    # --- Аугментации (настроены под дорожную разметку) ---
    hsv_h=0.015,            # лёгкая вариация оттенка (износ разметки)
    hsv_s=0.5,              # насыщенность (мокрая/сухая дорога)
    hsv_v=0.3,              # яркость (день/тень)
    degrees=5.0,            # лёгкий поворот (камера не идеально ровная)
    translate=0.1,          # сдвиг
    scale=0.3,              # масштаб (разная дистанция до разметки)
    flipud=0.0,             # НЕ переворачиваем вверх-вниз (дорога всегда снизу!)
    fliplr=0.5,             # горизонтальное зеркало — ок для разметки
    mosaic=1.0,             # mosaic — помогает при малом датасете и дисбалансе
    mixup=0.1,              # лёгкий mixup
    copy_paste=0.1,         # copy-paste аугментация (помогает редким классам)
    
    # --- Оптимизатор ---
    optimizer="AdamW",
    lr0=0.001,              # начальная скорость обучения
    lrf=0.01,               # финальная lr = lr0 * lrf
    weight_decay=0.0005,
    warmup_epochs=5,        # прогрев (важно для малого датасета)
    
    # --- Сохранение ---
    project=str(PROJECT_ROOT / "03_models"),
    name="yolov8n_seg_horizontal_v2",
    save=True,
    save_period=25,         # чекпоинт каждые 25 эпох
    plots=True,             # confusion matrix, loss curves
    
    # --- Прочее ---
    verbose=True,
    exist_ok=True,          # перезаписать если папка уже есть
)

print("Конфигурация обучения:")
for k, v in TRAIN_CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ### ⚡ Запуск обучения
# 
# Следующая ячейка запустит обучение. На AMD RX 6600M с batch=8, imgsz=640, 100 эпох — ожидай ~30-90 минут.
# 
# **Если вылетает OOM (Out of Memory по RAM):**
# 1. Уменьши `batch` → 4 или 2
# 2. Уменьши `imgsz` → 480
# 3. Уменьши `workers` → 0

# %%
# ОБУЧЕНИЕ — запуск!
results = model.train(**TRAIN_CONFIG)

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)

# %% [markdown]
# ## 5. Анализ результатов

# %%
# Путь к результатам обучения
run_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"]

print(f"Результаты в: {run_dir}")
print(f"\nСодержимое:")
for f in sorted(run_dir.iterdir()):
    size = f.stat().st_size // 1024 if f.is_file() else ""
    print(f"  {'📁' if f.is_dir() else '📄'} {f.name} {f'{size} KB' if size else ''}")

# %%
# Графики обучения (loss curves + метрики)
results_img = run_dir / "results.png"
if results_img.exists():
    img = cv2.imread(str(results_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Training Results — Loss & Metrics")
    plt.show()
else:
    print("results.png не найден (обучение ещё не завершено?)")

# %%
# Confusion Matrix
cm_img = run_dir / "confusion_matrix_normalized.png"
if cm_img.exists():
    img = cv2.imread(str(cm_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 14))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Normalized Confusion Matrix")
    plt.show()
else:
    print("confusion_matrix_normalized.png не найден")

# %% [markdown]
# ## 6. Валидация лучшей модели на test-сплите

# %%
# Загружаем лучшую модель
best_model_path = run_dir / "weights" / "best.pt"
if best_model_path.exists():
    best_model = YOLO(str(best_model_path))
    
    # Валидация на test-сплите
    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=TRAIN_CONFIG["device"],
        plots=True,
        save_json=True,
        project=str(run_dir / "test_eval"),
        name="test_results",
        exist_ok=True,
    )
    
    print("\n=== Метрики на TEST-сплите ===")
    print(f"mAP@50:      {test_metrics.seg.map50:.4f}")
    print(f"mAP@50-95:   {test_metrics.seg.map:.4f}")
    
    # Per-class mAP
    print("\nPer-class mAP@50:")
    for i, ap50 in enumerate(test_metrics.seg.ap50):
        name = data_cfg["names"].get(i, f"class_{i}")
        bar = "█" * int(ap50 * 30)
        print(f"  [{i:>2}] {name:<25} {ap50:.3f} {bar}")
else:
    print(f"best.pt не найден в {best_model_path}")
    print("Сначала запусти обучение (ячейка выше)")

# %% [markdown]
# ## 7. Визуализация предсказаний на test-изображениях

# %%
if best_model_path.exists():
    test_imgs = sorted((DATASET_DIR / "images" / "test").glob("*.*"))
    sample = random.sample(test_imgs, min(6, len(test_imgs)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, img_path in zip(axes.flat, sample):
        preds = best_model.predict(
            str(img_path), imgsz=640, conf=0.25,
            device=TRAIN_CONFIG["device"], verbose=False
        )
        # Отрисовка результата
        annotated = preds[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")
    
    plt.suptitle("Предсказания модели на TEST-изображениях", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "06_reports" / "test_predictions.png"), dpi=150)
    plt.show()
    print("Сохранено в 06_reports/test_predictions.png")

# %% [markdown]
# ## 8. Экспорт лучшей модели

# %%
if best_model_path.exists():
    import shutil
    
    # Копируем best.pt в 03_models/ с понятным именем
    export_path = MODELS_DIR / "yolov8n_seg_gost_best.pt"
    shutil.copy2(best_model_path, export_path)
    print(f"✅ Модель скопирована: {export_path}")
    print(f"   Размер: {export_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Экспорт в ONNX (для будущего пайплайна IPM)
    best_model.export(format="onnx", imgsz=640, simplify=True)
    onnx_src = run_dir / "weights" / "best.onnx"
    if onnx_src.exists():
        onnx_dst = MODELS_DIR / "yolov8n_seg_gost_best.onnx"
        shutil.copy2(onnx_src, onnx_dst)
        print(f"✅ ONNX экспортирован: {onnx_dst}")

# %% [markdown]
# ---
# ## Итого
# 
# После выполнения всех ячеек у тебя будет:
# 1. `03_models/yolov8n_seg_gost_v1/weights/best.pt` — лучший чекпоинт
# 2. `03_models/yolov8n_seg_gost_best.pt` — копия для удобства
# 3. `03_models/yolov8n_seg_gost_best.onnx` — для inference-пайплайна
# 4. `06_reports/train_samples_preview.png` — превью разметки
# 5. `06_reports/test_predictions.png` — визуализация предсказаний
# 
# **Следующий шаг:** модуль IPM (гомография → метры) для извлечения метрической геометрии разметки.

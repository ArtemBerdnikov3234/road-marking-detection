# %% [markdown]
# #  Обучение YOLOv8n-seg — Горизонтальная дорожная разметка
#
# **Датасет:** `02_yolo_dataset_horizontal` (26 классов: 1.12–1.26, ШП, 1.21)
# **Модель:** YOLOv8n-seg (nano segmentation, pretrained COCO)
# **Device:** CPU (AMD Ryzen 5 5600H)
#
# ---
# **Ячейки запускать по порядку сверху вниз (Ctrl+Enter).**

# %% [markdown]
# ## 1⃣ Окружение и пути

# %%
import sys
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(r"E:\project\SibDor")
DATASET_DIR  = PROJECT_ROOT / "02_yolo_dataset_horizontal"
DATA_YAML    = DATASET_DIR  / "data.yaml"
MODELS_DIR   = PROJECT_ROOT / "03_models"
REPORTS_DIR  = PROJECT_ROOT / "06_reports"

print(f"Python:  {sys.version}")
print(f"Датасет: {DATASET_DIR}")

# Читаем data.yaml
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

num_classes = len(data_cfg["names"])
print(f"Классов: {num_classes}")
print(f"Классы:  {data_cfg['names']}")

# %% [markdown]
# ## 2⃣ Проверка датасета

# %%
for split in ["train", "val", "test"]:
    imgs = list((DATASET_DIR / "images" / split).glob("*.*"))
    lbls = list((DATASET_DIR / "labels" / split).glob("*.txt"))
    status = "" if len(imgs) == len(lbls) and len(imgs) > 0 else ""
    print(f"  {split}: {len(imgs)} images, {len(lbls)} labels {status}")

# Подсчёт объектов по классам в train
from collections import Counter
train_class_counts = Counter()
for label_file in (DATASET_DIR / "labels" / "train").glob("*.txt"):
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            train_class_counts[int(line.split()[0])] += 1

print(f"\nОбъектов в train по классам:")
for cid in sorted(train_class_counts):
    name = data_cfg["names"].get(cid, f"id{cid}")
    cnt = train_class_counts[cid]
    bar = "" * min(cnt, 40)
    print(f"  [{cid:>2}] {name:<10} {cnt:>4} {bar}")

# %% [markdown]
# ## 3⃣ Превью train-изображений с разметкой

# %%
COLORS = [
    (255, 0, 85), (0, 255, 170), (85, 170, 255), (255, 170, 0),
    (170, 0, 255), (0, 255, 85), (255, 85, 170), (85, 255, 0),
    (170, 255, 85), (255, 0, 170), (0, 85, 255), (85, 0, 255),
]

train_imgs = sorted((DATASET_DIR / "images" / "train").glob("*.*"))
sample_imgs = random.sample(train_imgs, min(6, len(train_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, img_path in zip(axes.flat, sample_imgs):
    img_arr = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        ax.set_title(f" {img_path.name}", fontsize=8)
        ax.axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    label_path = DATASET_DIR / "labels" / "train" / f"{img_path.stem}.txt"
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            pts = np.array([(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)], dtype=np.int32)
            color = COLORS[cls_id % len(COLORS)]
            cv2.fillPoly(img, [pts], color=(*color, 80))
            cv2.polylines(img, [pts], True, color, 2)

    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=7)
    ax.axis("off")

plt.suptitle("Примеры train-изображений с разметкой", fontsize=14)
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "train_samples_preview.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 4⃣ Конфигурация обучения
#
# **Меняй параметры здесь перед запуском обучения (ячейка 5).**

# %%
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ========= НАСТРОЙКИ — МЕНЯТЬ ТУТ =========

MODEL_NAME    = "yolov8n-seg"                       # архитектура (n=nano, s=small, m=medium)
PRETRAINED    = str(MODELS_DIR / "yolov8n-seg.pt")  # pretrained веса
RUN_NAME      = "yolov8n_seg_horizontal_v3"         # имя эксперимента (папка в 03_models/)

TRAIN_CONFIG = dict(
    data=str(DATA_YAML),

    # --- Основные ---
    epochs=100,
    imgsz=1280,             # 1280 — мелкие стрелки видны лучше
    batch=4,                # уменьшено из-за imgsz=1280
    patience=30,            # early stopping
    workers=4,              # на Windows иногда нужно 0

    # --- Устройство ---
    device="cpu",           # Ultralytics не поддерживает DirectML

    # --- Аугментации ---
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    flipud=0.0,             # НЕ вверх-вниз (дорога всегда снизу!)
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    # --- Оптимизатор ---
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,

    # --- Сохранение ---
    project=str(MODELS_DIR),
    name=RUN_NAME,
    save=True,
    save_period=25,
    plots=True,
    verbose=True,
    exist_ok=True,
)

# ========= КОНЕЦ НАСТРОЕК =========

model = YOLO(PRETRAINED)
print(f"Модель:     {MODEL_NAME}")
print(f"Параметров: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
print(f"Эксперимент: {RUN_NAME}")
print(f"\nКонфигурация:")
for k, v in TRAIN_CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 5⃣ ОБУЧЕНИЕ
#
#  На CPU (imgsz=1280, batch=4, 100 эпох) — ожидай **~3-8 часов**.
#
# Если OOM → уменьши `batch` до 2 или `imgsz` до 640.

# %%
results = model.train(**TRAIN_CONFIG)
print("\n Обучение завершено!")

# Путь к результатам
RUN_DIR = MODELS_DIR / RUN_NAME
print(f"Результаты: {RUN_DIR}")

# %% [markdown]
# ## 6⃣ Графики обучения (Loss + Metrics)
#
# Эта и все следующие ячейки можно запускать **после** завершения обучения,
# даже в другой сессии — данные берутся из `results.csv`.

# %%
# Автоматически определяем RUN_DIR если запущено в новой сессии
if "RUN_DIR" not in dir() or not RUN_DIR.exists():
    RUN_DIR = MODELS_DIR / RUN_NAME
    print(f"[INFO] RUN_DIR = {RUN_DIR}")

# Загрузка results.csv
csv_path = RUN_DIR / "results.csv"
assert csv_path.exists(), f"Не найден {csv_path} — сначала запусти обучение (ячейка 5)"

with open(csv_path, "r") as f:
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]

epochs = [int(r["epoch"]) for r in rows]
print(f"Эпох в results.csv: {len(rows)}")

# Найти лучшую эпоху
best = max(rows, key=lambda r: float(r.get("metrics/mAP50(M)", "0")))
last = rows[-1]

print(f"\n{'Метрика':<25} {'Best (ep.' + best['epoch'] + ')':<18} {'Last (ep.' + last['epoch'] + ')':<18}")
print("-" * 55)
for name, key in [
    ("Precision (Mask)", "metrics/precision(M)"),
    ("Recall (Mask)",    "metrics/recall(M)"),
    ("mAP@50 (Mask)",   "metrics/mAP50(M)"),
    ("mAP@50-95 (Mask)","metrics/mAP50-95(M)"),
]:
    bv = float(best.get(key, "0"))
    lv = float(last.get(key, "0"))
    print(f"{name:<25} {bv:<18.4f} {lv:<18.4f}")

total_time = float(last.get("time", "0"))
print(f"\nВремя обучения: {total_time/3600:.1f} ч ({total_time:.0f} сек)")

# %%
# --- Loss curves ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Training & Validation Loss", fontsize=16, fontweight="bold")

loss_keys = [
    ("train/box_loss", "Box Loss", "#2196F3"),
    ("train/seg_loss", "Seg Loss", "#4CAF50"),
    ("train/cls_loss", "Cls Loss", "#FF9800"),
    ("train/dfl_loss", "DFL Loss", "#9C27B0"),
    ("val/box_loss",   "Val Box Loss", "#2196F3"),
    ("val/seg_loss",   "Val Seg Loss", "#4CAF50"),
    ("val/cls_loss",   "Val Cls Loss", "#FF9800"),
    ("val/dfl_loss",   "Val DFL Loss", "#9C27B0"),
]

for ax, (key, title, color) in zip(axes.flat, loss_keys):
    vals = [float(r.get(key, "0")) for r in rows]
    ax.plot(epochs, vals, color=color, lw=1.5, alpha=0.5)
    if len(vals) > 5:
        smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
        ax.plot(epochs[2:len(smooth)+2], smooth, color=color, lw=2.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "model_report" / "01_loss_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- Metric curves with max annotation ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mask Segmentation Metrics", fontsize=16, fontweight="bold")

metric_keys = [
    ("metrics/precision(M)", "Precision",  "#E91E63"),
    ("metrics/recall(M)",    "Recall",     "#00BCD4"),
    ("metrics/mAP50(M)",     "mAP@50",    "#4CAF50"),
    ("metrics/mAP50-95(M)",  "mAP@50-95", "#FF5722"),
]

for ax, (key, title, color) in zip(axes.flat, metric_keys):
    vals = [float(r.get(key, "0")) for r in rows]
    ax.plot(epochs, vals, color=color, lw=1.2, alpha=0.5, label="raw")
    if len(vals) > 5:
        smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
        ax.plot(epochs[2:len(smooth)+2], smooth, color=color, lw=2.5, label="smooth")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(title)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    # Аннотация максимума
    mx = max(vals); me = epochs[vals.index(mx)]
    ax.axhline(y=mx, color=color, ls="--", alpha=0.3)
    ax.annotate(f"max={mx:.3f}\n(ep.{me})", xy=(me, mx), fontsize=9,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "model_report" / "02_metrics_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- Dashboard ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Training Summary Dashboard", fontsize=16, fontweight="bold")

for ax, (key, title, color) in zip(axes, metric_keys):
    vals = [float(r.get(key, "0")) for r in rows]
    ax.fill_between(epochs, vals, alpha=0.2, color=color)
    ax.plot(epochs, vals, color=color, lw=2)
    ax.set_title(f"{title}\nBest: {max(vals):.3f}", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "model_report" / "03_dashboard.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7⃣ Confusion Matrix

# %%
cm_img = RUN_DIR / "confusion_matrix_normalized.png"
if cm_img.exists():
    img = cv2.imread(str(cm_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 14))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.show()
else:
    print("confusion_matrix_normalized.png не найден")

# %% [markdown]
# ## 8⃣ Per-class метрики (валидация best.pt)

# %%
best_pt = RUN_DIR / "weights" / "best.pt"
assert best_pt.exists(), f"best.pt не найден: {best_pt}"

best_model = YOLO(str(best_pt))

print("Запускаю валидацию на val-сплите...")
val_metrics = best_model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=1280,
    batch=4,
    device="cpu",
    verbose=False,
    plots=False,
)

ap50 = val_metrics.seg.ap50
class_names = [data_cfg["names"].get(i, f"cls_{i}") for i in range(len(ap50))]

print(f"\n{'Класс':<15} {'mAP@50':>8}  {'Статус'}")
print("-" * 45)
for i, name in enumerate(class_names):
    v = ap50[i]
    if v >= 0.7:    status = "🟢 Хорошо"
    elif v >= 0.3:  status = "🟡 Средне"
    elif v > 0:     status = " Плохо"
    else:           status = " Нет данных"
    bar = "" * int(v * 20)
    print(f"{name:<15} {v:>8.3f}  {bar:<20} {status}")

print(f"\nOverall mAP@50 (Mask): {float(np.mean(ap50)):.3f}")
print(f"Overall mAP@50-95:    {val_metrics.seg.map:.3f}")

# %%
# --- Per-class mAP@50 bar chart ---
colors_bar = []
for v in ap50:
    if v >= 0.7:    colors_bar.append("#4CAF50")
    elif v >= 0.3:  colors_bar.append("#FF9800")
    elif v > 0:     colors_bar.append("#FF5722")
    else:           colors_bar.append("#9E9E9E")

fig, ax = plt.subplots(figsize=(14, max(8, len(class_names) * 0.4)))
y_pos = np.arange(len(class_names))
bars = ax.barh(y_pos, ap50, color=colors_bar, height=0.7, edgecolor="white")

for bar, val in zip(bars, ap50):
    if val > 0:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(class_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("mAP@50", fontsize=12)
ax.set_title("Per-class mAP@50 (Mask Segmentation)", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.axvline(x=0.5, color="gray", ls="--", alpha=0.5, label="mAP=0.50")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "model_report" / "04_per_class_mAP50.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9⃣ Визуализация предсказаний (val + train)

# %%
def show_predictions(split: str, n_samples: int = 6):
    """Показывает предсказания модели на случайных изображениях из сплита."""
    imgs_dir = DATASET_DIR / "images" / split
    img_files = sorted([f for f in imgs_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    sample = random.sample(img_files, min(n_samples, len(img_files)))

    ncols = 3
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    axes_flat = np.array(axes).flat

    for ax, img_path in zip(axes_flat, sample):
        res = best_model.predict(str(img_path), imgsz=1280, conf=0.25, device="cpu", verbose=False)
        annotated = res[0].plot(line_width=2, font_size=10)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.set_title(img_path.name, fontsize=7)
        ax.axis("off")

    # Скрыть пустые ячейки
    for ax in list(axes_flat)[len(sample):]:
        ax.axis("off")

    plt.suptitle(f"Предсказания модели — {split} ({len(sample)} изображений)", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# Предсказания на val
show_predictions("val", n_samples=6)

# %%
# Предсказания на train
show_predictions("train", n_samples=6)

# %% [markdown]
# ##  Полный прогон предсказаний (val + train → файлы)
#
# Сохраняет аннотированные изображения в `06_reports/predictions_all/`.

# %%
import shutil

PRED_OUTPUT = REPORTS_DIR / "predictions_all"
if PRED_OUTPUT.exists():
    shutil.rmtree(PRED_OUTPUT)

total_saved = 0
for split in ["train", "val"]:
    imgs_dir = DATASET_DIR / "images" / split
    out_dir = PRED_OUTPUT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in imgs_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"--- {split}: {len(img_files)} изображений ---")

    for i, img_path in enumerate(img_files):
        res = best_model.predict(str(img_path), imgsz=1280, conf=0.25, device="cpu", verbose=False)
        annotated = res[0].plot(line_width=2, font_size=10)
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        total_saved += 1
        if (i+1) % 20 == 0 or (i+1) == len(img_files):
            print(f"  [{i+1}/{len(img_files)}]")

print(f"\n {total_saved} предсказаний сохранено в {PRED_OUTPUT}")

# %% [markdown]
# ## 1⃣1⃣ Экспорт модели

# %%
import shutil as _shutil

# Копирование best.pt
src_best = RUN_DIR / "weights" / "best.pt"
dst_best = MODELS_DIR / "yolov8n_seg_gost_best.pt"
_shutil.copy2(src_best, dst_best)
print(f" best.pt → {dst_best}  ({dst_best.stat().st_size / 1e6:.1f} MB)")

# ONNX экспорт
export_model = YOLO(str(src_best))
export_model.export(format="onnx", imgsz=1280, simplify=True)

onnx_src = src_best.with_suffix(".onnx")
onnx_dst = MODELS_DIR / "yolov8n_seg_gost_best.onnx"
if onnx_src.exists():
    _shutil.copy2(onnx_src, onnx_dst)
    print(f" ONNX → {onnx_dst}  ({onnx_dst.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 1⃣2⃣ Разметка видео
#
# Укажи путь к видеофайлу — модель обработает каждый кадр и сохранит
# размеченное видео в `06_reports/`.

# %%
# ========= УКАЖИ ПУТЬ К ВИДЕО =========
VIDEO_PATH = r"E:\project\SibDor\test_video.mp4"   # ← ПОМЕНЯЙ НА СВОЙ ФАЙЛ
CONF_VIDEO = 0.25                                   # порог уверенности (0.15–0.5)
IMGSZ_VIDEO = 1280                                  # разрешение inference
# =======================================

from pathlib import Path
import cv2
from ultralytics import YOLO

video_path = Path(VIDEO_PATH)
assert video_path.exists(), f"Видео не найдено: {video_path}"

# Загрузка модели (если ещё не загружена)
if "best_model" not in dir():
    best_pt = RUN_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = MODELS_DIR / "yolov8n_seg_gost_best.pt"
    best_model = YOLO(str(best_pt))
    print(f"Модель загружена: {best_pt.name}")

# Открываем видео
cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), f"Не удалось открыть видео: {video_path}"

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"Видео:     {video_path.name}")
print(f"Размер:    {w}x{h}")
print(f"FPS:       {fps:.1f}")
print(f"Кадров:    {total_frames}")
print(f"Длительность: {duration:.1f} сек ({duration/60:.1f} мин)")

# Выходной файл
out_name = f"{video_path.stem}_annotated.mp4"
out_path = REPORTS_DIR / out_name
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

print(f"\nОбработка → {out_path}")
print(f"Confidence: {CONF_VIDEO}, imgsz: {IMGSZ_VIDEO}\n")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = best_model.predict(
        frame,
        imgsz=IMGSZ_VIDEO,
        conf=CONF_VIDEO,
        device="cpu",
        verbose=False,
    )

    # Наложение масок и боксов
    annotated = results[0].plot(line_width=2, font_size=10)

    writer.write(annotated)
    frame_idx += 1

    if frame_idx % 50 == 0 or frame_idx == total_frames:
        pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
        print(f"  [{frame_idx}/{total_frames}] {pct:.0f}%")

cap.release()
writer.release()

file_size = out_path.stat().st_size / 1e6
print(f"\n Готово! Размеченное видео сохранено:")
print(f"   {out_path}")
print(f"   Размер: {file_size:.1f} MB")
print(f"   Кадров обработано: {frame_idx}")

# %% [markdown]
# ---
# ##  Готово!
#
# **Файлы модели:**
# - `03_models/yolov8n_seg_gost_best.pt` — для inference в Python
# - `03_models/yolov8n_seg_gost_best.onnx` — для деплоя
#
# **Следующий шаг:** IPM (Inverse Perspective Mapping) для перевода пикселей в метры.

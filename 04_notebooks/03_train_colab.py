# %% [markdown]
# #  YOLO26s — Детекция дорожной разметки (Google Colab)
#
# **Горизонтальная дорожная разметка ГОСТ Р 51256-2018**
#
# ## Инструкция:
# 1. Загрузи `dataset_horizontal.zip` на Google Drive (в корень «Мой Диск»)
# 2. Открой в Colab: File → Open in Colab
# 3. Включи GPU: Runtime → Change runtime type → T4 GPU
# 4. Запускай ячейки по порядку (Shift+Enter)

# %% [markdown]
# ## 1⃣ Подключение Google Drive + установка пакетов

# %%
from google.colab import drive   # pyright: ignore [reportMissingImports]
drive.mount("/content/drive")

# %%
!pip install -q ultralytics  # pyright: ignore [reportMissingImports]

# %% [markdown]
# ## 2⃣ Распаковка и нормализация датасета

# %%
import os
import shutil
import yaml
from pathlib import Path

DRIVE_ZIP = Path("/content/drive/MyDrive/horizontal_dataset.zip")
DATASET_DIR = Path("/content/Dataset")

assert DRIVE_ZIP.exists(), f"Не найден {DRIVE_ZIP} — загрузи horizontal_dataset.zip на Google Drive в корень"

if DATASET_DIR.exists():
    print(f"Папка {DATASET_DIR} уже существует. Удаляем старую версию, чтобы распаковать новую...")
    shutil.rmtree(DATASET_DIR)

print("Распаковка датасета...")
shutil.unpack_archive(str(DRIVE_ZIP), str(DATASET_DIR))
print("Распаковка завершена.")

# %% [markdown]
# ## 3⃣ Фикс путей в data.yaml

# %%
DATA_YAML = DATASET_DIR / "data.yaml"

if DATA_YAML.exists():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Указываем абсолютный путь к датасету внутри Colab
    data_cfg["path"] = str(DATASET_DIR)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Файл data.yaml успешно настроен для Colab! Путь: {data_cfg['path']}")
    print(f"Классов: {len(data_cfg['names'])}")
else:
    print("ВНИМАНИЕ: файл data.yaml не найден в корне датасета!")

# %% [markdown]
# ## 3.1⃣ Анализ классов датасета — количество объектов по сплитам
#
# Подробная статистика: сколько объектов каждого класса в train/val/test.
# Позволяет сразу увидеть дисбаланс и редкие классы.

# %%
from collections import Counter
import numpy as np

def count_objects_per_class(labels_dir):
    """Считает количество объектов каждого класса в папке с метками."""
    counter = Counter()
    num_files = 0
    if not labels_dir.exists():
        return counter, 0
    for label_file in labels_dir.glob("*.txt"):
        num_files += 1
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counter[int(line.split()[0])] += 1
    return counter, num_files

print("=" * 70)
print(" АНАЛИЗ ДАТАСЕТА: классы и количество объектов")
print("=" * 70)

split_counters = {}
split_files = {}
for split in ["train", "val", "test"]:
    counter, nf = count_objects_per_class(DATASET_DIR / "labels" / split)
    split_counters[split] = counter
    split_files[split] = nf

# Шапка таблицы
header = f"{'id':>3}  {'Класс':<12} {'train':>7} {'val':>7} {'test':>7} {'ВСЕГО':>7}   Проблема"
print(f"\n{header}")
print("-" * len(header))

all_class_ids = sorted(data_cfg["names"].keys())
total_objects = 0
rare_classes = []
missing_in_train = []

for cid in all_class_ids:
    name = data_cfg["names"][cid]
    tr = split_counters["train"].get(cid, 0)
    va = split_counters["val"].get(cid, 0)
    te = split_counters["test"].get(cid, 0)
    total = tr + va + te
    total_objects += total

    # Определяем проблемы
    flags = []
    if total == 0:
        flags.append(" нет в датасете")
    else:
        if tr == 0:
            flags.append(" НЕТ В TRAIN")
            missing_in_train.append(name)
        if va == 0:
            flags.append("нет в val")
        if te == 0:
            flags.append("нет в test")
        if total < 20:
            flags.append(" РЕДКИЙ")
            rare_classes.append((cid, name, total))

    flag_str = "; ".join(flags)
    print(f"{cid:>3}  {name:<12} {tr:>7} {va:>7} {te:>7} {total:>7}   {flag_str}")

print(f"\n{'' * 70}")
print(f"Файлов:   train={split_files['train']}, val={split_files['val']}, test={split_files['test']}")
print(f"Объектов: train={sum(split_counters['train'].values())}, "
      f"val={sum(split_counters['val'].values())}, "
      f"test={sum(split_counters['test'].values())}")
print(f"Всего объектов: {total_objects}")
print(f"Классов в датасете: {len([c for c in all_class_ids if sum(split_counters[s].get(c,0) for s in ['train','val','test']) > 0])} "
      f"из {len(all_class_ids)}")

if rare_classes:
    print(f"\n  Редких классов (<20 объектов): {len(rare_classes)}")
    for cid, name, cnt in rare_classes:
        print(f"    id={cid} {name}: {cnt} объектов")
if missing_in_train:
    print(f"\n Классы без примеров в train: {missing_in_train}")
    print("   Модель НЕ сможет научиться их распознавать!")
print("=" * 70)

# %% [markdown]
# ##  Oversampling редких классов
#
# **Oversampling выполняется ЛОКАЛЬНО** перед загрузкой датасета на Google Drive.
# Используй скрипт `05_scripts/05_oversample_rare_classes.py`:
#
# ```bash
# cd 05_scripts
# python 05_oversample_rare_classes.py --dataset ../02_yolo_dataset_horizontal --target 80 --max-dupes 5
# ```
#
# После этого заново запакуй датасет в `dataset_horizontal.zip` и загрузи на Drive.
# Подробности алгоритма — в самом скрипте.

# %% [markdown]
# ## 4⃣ Проверка GPU

# %%
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(" GPU не найден! Включи: Runtime → Change runtime type → T4 GPU")

# %% [markdown]
# ## 5⃣ Конфигурация обучения
#
# **Ключевые изменения:**
# - YOLO26s (детекция) — новейшая архитектура Ultralytics (янв. 2026)
#   NMS-free, DFL-free head, Progressive Loss для мелких объектов
# - patience=40 — early stopping с запасом для малых датасетов
# - 300 эпох максимум, cosine LR schedule
# - batch=16 — детекция легче по памяти, чем сегментация

# %%
from ultralytics import YOLO  # pyright: ignore [reportMissingImports]

import datetime
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"yolo26s_det_horizontal_{_timestamp}"
MODELS_DIR = Path("/content/models")

# YOLO26s — детекция (не сегментация!)
model = YOLO("yolo26s.pt")

# Определяем кол-во картинок в train (с учётом oversampling)
_train_img_count = len(list((DATASET_DIR / "images" / "train").glob("*.*")))
_is_small_dataset = _train_img_count < 500
print(f"Train images: {_train_img_count} {'(малый датасет — усиленная регуляризация)' if _is_small_dataset else ''}")

TRAIN_CONFIG = dict(
    data=str(DATA_YAML),

    # --- Основные ---
    epochs=300,             # Больше эпох для малых датасетов (early stopping остановит)
    imgsz=1280,             # Высокое разрешение для мелкой разметки
    batch=-1,               # Авто: максимальный batch под GPU Colab
    cache="ram",             # Кеширование в RAM -- ускоряет обучение в Colab
    patience=40,            #  Увеличен: малым датасетам нужно больше терпения
    workers=2,
    close_mosaic=20,        # Отключить мозаику за 20 эпох до конца

    # --- Устройство ---
    device=0,

    # --- Аугментации (усилены для малого датасета дорожной разметки) ---
    hsv_h=0.015,            # Лёгкий сдвиг тона
    hsv_s=0.5,              # Насыщенность — важно для разных погодных условий
    hsv_v=0.5,              # Яркость — тени/освещение на дороге
    degrees=5.0,            # Умеренный поворот (камера может быть не идеально ровной)
    translate=0.2,          # Сдвиг — разметка может быть в разных частях кадра
    scale=0.5,              # Масштаб — разметка бывает близко/далеко
    shear=3.0,              # Перспектива — вид с камеры под углом
    perspective=0.0005,     # Лёгкая перспективная деформация
    flipud=0.0,             # Отключено: верх/низ важен для разметки
    fliplr=1.0,             # Горизонтальное отражение всегда (дорога симметрична)
    mosaic=1.0,             # Мозаика — ключевая аугментация для малых датасетов
    mixup=0.2,              # Mixup усилен для разнообразия
    copy_paste=0.1,         # Copy-paste: копирует объекты между изображениями
    erasing=0.35,           # Random erasing — борьба с переобучением
    crop_fraction=0.75,     # Случайный кроп — модель видит фрагменты

    # --- Оптимизатор (оптимизирован для малых датасетов) ---
    optimizer="AdamW",
    lr0=0.0008,             # Чуть ниже LR — стабильнее на малых данных
    lrf=0.01,               # Финальный LR = lr0 * lrf
    cos_lr=True,            # Cosine annealing (плавное затухание)
    weight_decay=0.001,     # Усиленная L2-регуляризация против переобучения
    warmup_epochs=8,        # Дольше прогрев — стабилизирует начало обучения
    warmup_momentum=0.5,    # Плавнее старт

    # --- Loss ---
    box=7.5,                # Box loss weight
    cls=1.5,                #  Classification loss — важнее для 26 классов
    dfl=1.5,                # Distribution focal loss weight

    # --- Dropout / регуляризация ---
    dropout=0.15,           # Dropout в head — борьба с переобучением

    # --- Сохранение ---
    project=str(MODELS_DIR),
    name=RUN_NAME,
    save=True,
    save_period=25,         # Сохранение чекпоинтов каждые 25 эпох
    plots=True,
    verbose=True,
    exist_ok=True,
)

print(f"Модель:      YOLO26s (Detection)")
print(f"Параметров:  {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
print(f"Задача:      Detection (bbox, NMS-free)")
print(f"Эксперимент: {RUN_NAME}")
print(f"Эпох:        {TRAIN_CONFIG['epochs']} (patience={TRAIN_CONFIG['patience']})")
print(f"Device:      {'GPU ' if torch.cuda.is_available() else 'CPU '}")

# %% [markdown]
# ## 6⃣ ОБУЧЕНИЕ
#
#  На T4 GPU — ожидай **~40-120 минут** (до 300 эпох, early stopping).
# Обучение остановится автоматически, когда mAP перестанет расти
# в течение 40 эпох подряд.

# %%
results = model.train(**TRAIN_CONFIG)

RUN_DIR = MODELS_DIR / RUN_NAME
print(f"\n Обучение завершено!")
print(f"Результаты: {RUN_DIR}")

# Сколько эпох реально обучалось
import csv
csv_path = RUN_DIR / "results.csv"
with open(csv_path, "r") as f:
    rows_count = sum(1 for _ in csv.DictReader(f))
print(f"Фактических эпох: {rows_count} из {TRAIN_CONFIG['epochs']}")
if rows_count < TRAIN_CONFIG['epochs']:
    print(f" Early stopping сработал на эпохе {rows_count}!")

# %% [markdown]
# ## 7⃣ Метрики и графики

# %%
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = RUN_DIR / "results.csv"
with open(csv_path, "r") as f:
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]

epochs = [int(r["epoch"]) for r in rows]

# Detection метрики (Box, не Mask!)
best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", "0")))
last = rows[-1]

print(f"\n{'Метрика':<25} {'Best (ep.' + best['epoch'] + ')':<18} {'Last (ep.' + last['epoch'] + ')':<18}")
print("-" * 60)
for name, key in [
    ("Precision (Box)",  "metrics/precision(B)"),
    ("Recall (Box)",     "metrics/recall(B)"),
    ("mAP@50 (Box)",    "metrics/mAP50(B)"),
    ("mAP@50-95 (Box)", "metrics/mAP50-95(B)"),
]:
    bv = float(best.get(key, "0"))
    lv = float(last.get(key, "0"))
    print(f"{name:<25} {bv:<18.4f} {lv:<18.4f}")

total_time = sum(float(r.get("time", "0")) for r in rows)
print(f"\nВремя обучения: {total_time/3600:.1f} ч ({total_time:.0f} сек)")
print(f"Всего эпох: {len(epochs)} (patience={TRAIN_CONFIG['patience']})")

# %%
# Loss curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

# Losses
for ax, (key, title, color) in zip(axes[0], [
    ("train/box_loss", "Box Loss",  "#E91E63"),
    ("train/cls_loss", "Cls Loss",  "#2196F3"),
    ("train/dfl_loss", "DFL Loss",  "#FF9800"),
]):
    vals = [float(r.get(key, "0")) for r in rows]
    ax.plot(epochs, vals, color=color, lw=1.2, alpha=0.5, label="raw")
    if len(vals) > 5:
        smooth = np.convolve(vals, np.ones(7)/7, mode="valid")
        ax.plot(epochs[3:len(smooth)+3], smooth, color=color, lw=2.5, label="smooth")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# Metrics
for ax, (key, title, color) in zip(axes[1], [
    ("metrics/precision(B)", "Precision",  "#E91E63"),
    ("metrics/mAP50(B)",     "mAP@50",    "#4CAF50"),
    ("metrics/mAP50-95(B)",  "mAP@50-95", "#FF5722"),
]):
    vals = [float(r.get(key, "0")) for r in rows]
    ax.plot(epochs, vals, color=color, lw=1.2, alpha=0.5, label="raw")
    if len(vals) > 5:
        smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
        ax.plot(epochs[2:len(smooth)+2], smooth, color=color, lw=2.5, label="smooth")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(title)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    mx = max(vals); me = epochs[vals.index(mx)]
    ax.axhline(y=mx, color=color, ls="--", alpha=0.3)
    ax.annotate(f"max={mx:.3f}\n(ep.{me})", xy=(me, mx), fontsize=9,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(str(RUN_DIR / "training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8⃣ Per-class метрики + Оценка на val и test

# %%
best_pt = RUN_DIR / "weights" / "best.pt"
best_model = YOLO(str(best_pt))

# --- Оценка на VAL ---
print("=" * 60)
print(" Оценка на VAL")
print("=" * 60)
val_metrics = best_model.val(
    data=str(DATA_YAML), split="val", imgsz=1280,
    batch=16, device=0, verbose=False, plots=True
)
ap50 = val_metrics.box.ap50
class_names = [data_cfg["names"].get(i, f"cls_{i}") for i in range(len(ap50))]

print(f"\n{'Класс':<15} {'mAP@50':>8}  {'Статус'}")
print("-" * 50)
for i, name in enumerate(class_names):
    v = ap50[i]
    if v >= 0.7:    status = "🟢 Хорошо"
    elif v >= 0.3:  status = "🟡 Средне"
    elif v > 0:     status = " Плохо"
    else:           status = " Нет данных"
    bar = "" * int(v * 20)
    print(f"{name:<15} {v:>8.3f}  {bar:<20} {status}")

overall_map50 = float(np.mean(ap50))
print(f"\nOverall val mAP@50: {overall_map50:.3f}")

# --- Оценка на TEST ---
print("\n" + "=" * 60)
print(" Оценка на TEST")
print("=" * 60)
test_metrics = best_model.val(
    data=str(DATA_YAML), split="test", imgsz=1280,
    batch=16, device=0, verbose=False, plots=False
)
test_ap50 = test_metrics.box.ap50

print(f"\n{'Класс':<15} {'Val mAP@50':>10} {'Test mAP@50':>12}  {'Разница':>8}")
print("-" * 55)
for i, name in enumerate(class_names):
    v_val = ap50[i]
    v_test = test_ap50[i] if i < len(test_ap50) else 0.0
    diff = v_test - v_val
    sign = "+" if diff >= 0 else ""
    print(f"{name:<15} {v_val:>10.3f} {v_test:>12.3f}  {sign}{diff:>7.3f}")

test_overall = float(np.mean(test_ap50))
print(f"\nOverall test mAP@50: {test_overall:.3f}")
print(f"Val→Test разница:    {test_overall - overall_map50:+.3f}")

# %%
# Per-class bar chart
colors_bar = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.3 else "#FF5722" if v > 0 else "#9E9E9E" for v in ap50]

fig, ax = plt.subplots(figsize=(14, max(8, len(class_names) * 0.4)))
y_pos = np.arange(len(class_names))
bars = ax.barh(y_pos, ap50, color=colors_bar, height=0.7, edgecolor="white")
for bar, val in zip(bars, ap50):
    if val > 0:
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(y_pos)
ax.set_yticklabels(class_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("mAP@50", fontsize=12)
ax.set_title("Per-class mAP@50 (Detection)", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.axvline(x=0.5, color="gray", ls="--", alpha=0.5)
ax.axvline(x=0.7, color="green", ls="--", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(str(RUN_DIR / "per_class_map50.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9⃣ Confusion Matrix

# %%
import cv2

cm_img = RUN_DIR / "confusion_matrix_normalized.png"
if cm_img.exists():
    img = cv2.imread(str(cm_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 16))
    plt.imshow(img); plt.axis("off")
    plt.title("Normalized Confusion Matrix (Detection)", fontsize=16)
    plt.show()
else:
    print("Confusion matrix не найдена — будет сгенерирована при val()")

# %% [markdown]
# ##  Скачивание модели на Google Drive

# %%
import shutil

SAVE_DIR_BASE = Path("/content/drive/MyDrive/SibDor_models")
SAVE_DIR = SAVE_DIR_BASE / RUN_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# best.pt
src = RUN_DIR / "weights" / "best.pt"
dst = SAVE_DIR / "yolov8s_det_horizontal_best.pt"
shutil.copy2(src, dst)
print(f" best.pt → {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")

# last.pt
src_last = RUN_DIR / "weights" / "last.pt"
if src_last.exists():
    dst_last = SAVE_DIR / "yolov8s_det_horizontal_last.pt"
    shutil.copy2(src_last, dst_last)
    print(f" last.pt → {dst_last}  ({dst_last.stat().st_size / 1e6:.1f} MB)")

# ONNX export
export_model = YOLO(str(src))
export_model.export(format="onnx", imgsz=1280, simplify=True, device=0)

onnx_src = src.with_suffix(".onnx")
if onnx_src.exists():
    onnx_dst = SAVE_DIR / "yolov8s_det_horizontal_best.onnx"
    shutil.copy2(onnx_src, onnx_dst)
    print(f" ONNX → {onnx_dst}  ({onnx_dst.stat().st_size / 1e6:.1f} MB)")

# results.csv
shutil.copy2(RUN_DIR / "results.csv", SAVE_DIR / "results.csv")
print(f" results.csv → {SAVE_DIR / 'results.csv'}")

# Графики
for plot_name in ["training_curves.png", "per_class_map50.png",
                  "confusion_matrix_normalized.png", "results.png"]:
    plot_src = RUN_DIR / plot_name
    if plot_src.exists():
        shutil.copy2(plot_src, SAVE_DIR / plot_name)

print(f"\n Все файлы на Drive: {SAVE_DIR}")
print("Теперь скачай папку SibDor_models с Google Drive на свой ПК.")

# %% [markdown]
# ## 1⃣1⃣ Сводный отчёт

# %%
print("=" * 60)
print(" СВОДНЫЙ ОТЧЁТ")
print("=" * 60)
print(f"Модель:        YOLOv8s (Detection)")
print(f"Параметров:    {sum(p.numel() for p in best_model.model.parameters()) / 1e6:.1f}M")
print(f"Датасет:       {len(data_cfg['names'])} классов")
print(f"Эпох:          {len(epochs)}/{TRAIN_CONFIG['epochs']}")
early = " (early stopping)" if len(epochs) < TRAIN_CONFIG['epochs'] else ""
print(f"               {early}")
print(f"Лучшая эпоха:  {best['epoch']}")
print(f"")
print(f"Val  mAP@50:   {overall_map50:.3f}")
print(f"Test mAP@50:   {test_overall:.3f}")
print(f"Val  P/R:      {float(best.get('metrics/precision(B)', 0)):.3f} / {float(best.get('metrics/recall(B)', 0)):.3f}")
print(f"")

good = sum(1 for v in ap50 if v >= 0.7)
medium = sum(1 for v in ap50 if 0.3 <= v < 0.7)
bad = sum(1 for v in ap50 if 0 < v < 0.3)
zero = sum(1 for v in ap50 if v <= 0)
print(f"Классы 🟢 ≥0.7: {good}")
print(f"Классы 🟡 0.3-0.7: {medium}")
print(f"Классы  <0.3: {bad}")
print(f"Классы  =0:   {zero}")
print("=" * 60)

# %% [markdown]
# ---
# ##  Готово!
#
# Файлы сохранены на Google Drive в папке `SibDor_models/`:
# - `yolov8s_det_horizontal_best.pt` — для inference
# - `yolov8s_det_horizontal_best.onnx` — для деплоя
# - `results.csv` — метрики по эпохам
# - `training_curves.png`, `per_class_map50.png` — визуализация
#
# **Скачай** `best.pt` с Drive и положи в `03_models/` на своём ПК.

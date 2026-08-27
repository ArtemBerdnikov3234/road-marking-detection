"""
06_model_report.py

Анализ обученной модели YOLOv8n-seg: метрики, графики, per-class статистика.
Сохраняет всё в 06_reports/model_report/.

Запуск:
    python 06_model_report.py
"""

import matplotlib
matplotlib.use("Agg")

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


# === Настройки ===
PROJECT_ROOT = Path(r"E:\project\SibDor")
RUN_DIR = PROJECT_ROOT / "03_models" / "yolov8n_seg_horizontal_v2"
DATASET_DIR = PROJECT_ROOT / "02_yolo_dataset_horizontal"
REPORT_DIR = PROJECT_ROOT / "06_reports" / "model_report"


def load_results_csv(run_dir: Path) -> list[dict]:
    """Загружает results.csv из директории обучения."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        sys.exit(f"[ОШИБКА] Не найден {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(clean)
    return rows


def load_class_names(dataset_dir: Path) -> dict:
    """Загружает имена классов из data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", {})


def print_model_summary(run_dir: Path):
    """Печатает информацию о модели из args.yaml."""
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        print("[ПРЕДУПРЕЖДЕНИЕ] args.yaml не найден")
        return

    with open(args_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    print("=" * 60)
    print("  МОДЕЛЬ: YOLOv8n-seg (horizontal markings)")
    print("=" * 60)
    print(f"  Архитектура:    {args.get('model', 'N/A')}")
    print(f"  Задача:         {args.get('task', 'N/A')}")
    print(f"  Эпохи:          {args.get('epochs', 'N/A')}")
    print(f"  Batch size:     {args.get('batch', 'N/A')}")
    print(f"  Image size:     {args.get('imgsz', 'N/A')}")
    print(f"  Optimizer:      {args.get('optimizer', 'N/A')}")
    print(f"  LR0:            {args.get('lr0', 'N/A')}")
    print(f"  Weight decay:   {args.get('weight_decay', 'N/A')}")
    print(f"  Device:         {args.get('device', 'N/A')}")
    print(f"  Patience:       {args.get('patience', 'N/A')}")
    print(f"  Mosaic:         {args.get('mosaic', 'N/A')}")
    print(f"  Copy-paste:     {args.get('copy_paste', 'N/A')}")


def print_metrics_table(rows: list[dict]):
    """Находит лучшую эпоху и печатает таблицу метрик."""
    best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(M)", "0")))
    last_row = rows[-1]

    print("\n" + "=" * 60)
    print("  МЕТРИКИ МОДЕЛИ")
    print("=" * 60)

    headers = ["Метрика", "Best (epoch " + best_row["epoch"] + ")", "Last (epoch " + last_row["epoch"] + ")"]
    metrics = [
        ("Precision (Box)", "metrics/precision(B)"),
        ("Recall (Box)", "metrics/recall(B)"),
        ("mAP@50 (Box)", "metrics/mAP50(B)"),
        ("mAP@50-95 (Box)", "metrics/mAP50-95(B)"),
        ("", ""),
        ("Precision (Mask)", "metrics/precision(M)"),
        ("Recall (Mask)", "metrics/recall(M)"),
        ("mAP@50 (Mask)", "metrics/mAP50(M)"),
        ("mAP@50-95 (Mask)", "metrics/mAP50-95(M)"),
    ]

    print(f"\n  {'Метрика':<25} {'Best (ep.' + best_row['epoch'] + ')':<18} {'Last (ep.' + last_row['epoch'] + ')':<18}")
    print("  " + "-" * 55)
    for name, key in metrics:
        if not key:
            print()
            continue
        best_val = float(best_row.get(key, "0"))
        last_val = float(last_row.get(key, "0"))
        print(f"  {name:<25} {best_val:<18.4f} {last_val:<18.4f}")

    # Время обучения
    total_time = float(last_row.get("time", "0"))
    hours = total_time / 3600
    print(f"\n  Время обучения:   {hours:.1f} часов ({total_time:.0f} сек)")

    return best_row


def plot_training_curves(rows: list[dict], report_dir: Path):
    """Рисует графики loss и метрик по эпохам."""
    epochs = [int(r["epoch"]) for r in rows]

    # --- Рисунок 1: Loss curves ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Training & Validation Loss", fontsize=16, fontweight="bold")

    loss_configs = [
        ("train/box_loss", "Box Loss", "#2196F3"),
        ("train/seg_loss", "Seg Loss", "#4CAF50"),
        ("train/cls_loss", "Cls Loss", "#FF9800"),
        ("train/dfl_loss", "DFL Loss", "#9C27B0"),
        ("val/box_loss", "Val Box Loss", "#2196F3"),
        ("val/seg_loss", "Val Seg Loss", "#4CAF50"),
        ("val/cls_loss", "Val Cls Loss", "#FF9800"),
        ("val/dfl_loss", "Val DFL Loss", "#9C27B0"),
    ]

    for ax, (key, title, color) in zip(axes.flat, loss_configs):
        values = [float(r.get(key, "0")) for r in rows]
        ax.plot(epochs, values, color=color, linewidth=1.5, alpha=0.7)
        # Сглаженная линия
        if len(values) > 5:
            smooth = np.convolve(values, np.ones(5) / 5, mode="valid")
            ax.plot(epochs[2:len(smooth) + 2], smooth, color=color, linewidth=2.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = report_dir / "01_loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.name}")

    # --- Рисунок 2: Метрики ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mask Segmentation Metrics", fontsize=16, fontweight="bold")

    metric_configs = [
        ("metrics/precision(M)", "Precision", "#E91E63"),
        ("metrics/recall(M)", "Recall", "#00BCD4"),
        ("metrics/mAP50(M)", "mAP@50", "#4CAF50"),
        ("metrics/mAP50-95(M)", "mAP@50-95", "#FF5722"),
    ]

    for ax, (key, title, color) in zip(axes.flat, metric_configs):
        values = [float(r.get(key, "0")) for r in rows]
        ax.plot(epochs, values, color=color, linewidth=1.2, alpha=0.5, label="raw")
        if len(values) > 5:
            smooth = np.convolve(values, np.ones(5) / 5, mode="valid")
            ax.plot(epochs[2:len(smooth) + 2], smooth, color=color, linewidth=2.5, label="smooth")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Отметить максимум
        max_val = max(values)
        max_epoch = epochs[values.index(max_val)]
        ax.axhline(y=max_val, color=color, linestyle="--", alpha=0.3)
        ax.annotate(f"max={max_val:.3f}\n(ep.{max_epoch})",
                    xy=(max_epoch, max_val), fontsize=9,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = report_dir / "02_metrics_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.name}")

    # --- Рисунок 3: Сводный dashboard ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Training Summary Dashboard", fontsize=16, fontweight="bold")

    for ax, (key, title, color) in zip(axes, metric_configs):
        values = [float(r.get(key, "0")) for r in rows]
        ax.fill_between(epochs, values, alpha=0.2, color=color)
        ax.plot(epochs, values, color=color, linewidth=2)
        max_val = max(values)
        ax.set_title(f"{title}\nBest: {max_val:.3f}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = report_dir / "03_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.name}")


def plot_per_class_analysis(run_dir: Path, class_names: dict, report_dir: Path):
    """Рисует per-class анализ из val результатов."""
    # Используем данные из val predictions (последний val при обучении)
    # Запускаем YOLO val для получения per-class метрик
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("  [SKIP] best.pt не найден, пропускаю per-class анализ")
        return

    model = YOLO(str(best_pt))
    data_yaml = DATASET_DIR / "data.yaml"

    print("\n  Запускаю валидацию для per-class метрик...")
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=1280,
        batch=4,
        device="cpu",
        verbose=False,
        plots=False,
    )

    # Per-class mAP@50
    ap50 = metrics.seg.ap50
    names = [class_names.get(i, f"cls_{i}") for i in range(len(ap50))]

    # --- Рисунок 4: Per-class mAP@50 ---
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = []
    for v in ap50:
        if v >= 0.7:
            colors.append("#4CAF50")  # зелёный — хорошо
        elif v >= 0.3:
            colors.append("#FF9800")  # оранжевый — средне
        elif v > 0:
            colors.append("#FF5722")  # красный — плохо
        else:
            colors.append("#9E9E9E")  # серый — нет данных

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, ap50, color=colors, height=0.7, edgecolor="white")

    # Подписи значений
    for bar, val in zip(bars, ap50):
        if val > 0:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("mAP@50", fontsize=12)
    ax.set_title("Per-class mAP@50 (Mask Segmentation) — Validation Set",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="mAP=0.50")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = report_dir / "04_per_class_mAP50.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.name}")

    # --- Рисунок 5: Per-class Precision vs Recall ---
    ap = metrics.seg.ap
    p = metrics.seg.p
    r = metrics.seg.r

    # p и r могут быть массивами по threshold, берём среднее
    if hasattr(p, '__len__') and len(p) > 0:
        if isinstance(p[0], (list, np.ndarray)):
            p_avg = [np.mean(x) if len(x) > 0 else 0 for x in p]
            r_avg = [np.mean(x) if len(x) > 0 else 0 for x in r]
        else:
            p_avg = list(p)
            r_avg = list(r)
    else:
        p_avg = [0] * len(names)
        r_avg = [0] * len(names)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, name in enumerate(names):
        if i < len(p_avg) and i < len(r_avg):
            px, rx = float(p_avg[i]), float(r_avg[i])
            color = colors[i] if i < len(colors) else "#9E9E9E"
            ax.scatter(rx, px, s=100, color=color, edgecolor="black", linewidth=0.5, zorder=5)
            ax.annotate(name, (rx, px), fontsize=8, ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision vs Recall per Class", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [1, 0], "k--", alpha=0.2)  # diagonal
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = report_dir / "05_precision_vs_recall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.name}")

    # Печать per-class таблицы
    print("\n" + "=" * 60)
    print("  PER-CLASS МЕТРИКИ (Mask mAP@50)")
    print("=" * 60)
    print(f"  {'Класс':<15} {'mAP@50':>8} {'Статус':<15}")
    print("  " + "-" * 40)
    for i, name in enumerate(names):
        val = ap50[i] if i < len(ap50) else 0
        if val >= 0.7:
            status = "🟢 Хорошо"
        elif val >= 0.3:
            status = "🟡 Средне"
        elif val > 0:
            status = "🔴 Плохо"
        else:
            status = "⚫ Нет данных"
        bar = "█" * int(val * 20)
        print(f"  {name:<15} {val:>8.3f} {bar:<20} {status}")

    overall_map50 = float(np.mean(ap50))
    print(f"\n  Overall mAP@50: {overall_map50:.3f}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные обучения...")
    rows = load_results_csv(RUN_DIR)
    class_names = load_class_names(DATASET_DIR)

    # 1. Информация о модели
    print_model_summary(RUN_DIR)

    # 2. Таблица метрик
    print_metrics_table(rows)

    # 3. Графики
    print("\n  Строю графики...")
    plot_training_curves(rows, REPORT_DIR)

    # 4. Per-class анализ
    plot_per_class_analysis(RUN_DIR, class_names, REPORT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  ВСЕ ОТЧЁТЫ СОХРАНЕНЫ В: {REPORT_DIR}")
    print(f"{'=' * 60}")
    print(f"  01_loss_curves.png        — графики loss")
    print(f"  02_metrics_curves.png     — precision/recall/mAP кривые")
    print(f"  03_dashboard.png          — сводная панель")
    print(f"  04_per_class_mAP50.png    — per-class mAP@50")
    print(f"  05_precision_vs_recall.png — P vs R scatter")


if __name__ == "__main__":
    main()

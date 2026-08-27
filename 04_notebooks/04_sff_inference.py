# %% [markdown]
# # Детекция дорожной разметки на SFF-видео
# YOLO + ByteTrack + border-exit. Batch (CSV) и Demo (видео) режимы.

# %%
import os, sys, glob, math, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
SFF_READER_DIR = r"C:\Users\Артём\Desktop\sibdor\работа с sff"
if SFF_READER_DIR not in sys.path:
    sys.path.append(SFF_READER_DIR)
from sff_reader import SFFReader

# %% [markdown]
# ## Настройки

# %%
MODEL_PATH = r"e:\project\SibDor\03_models\yolov8s_det_horizontal_best.pt"
VIDEOS_DIR = r"Camera_2"
SFF_GLOB_PATTERN = "**/Video.sff"
BASE_OUTPUT_DIR = Path(r"e:\project\SibDor\06_inference_results")

CONF_THRESHOLD = 0.25
CLASSES_FILTER = None
TRACKER_CFG = "bytetrack.yaml"
TARGET_AREA = 1024 * 1024
VIDEO_FPS = 10
MIN_BOX_AREA = 500

BORDER_MARGIN_PX = 50
BORDER_MARGIN_PERCENT = 0.20
BORDER_SIDES = {"bottom", "left", "right"}
DEBUG_BORDER_EXITS = True

CONF_SNAPSHOT_MIN = 0.50
SNAPSHOT_PICKET_GAP = 30
MERGE_PICKET_RADIUS = 30

_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% [markdown]
# ## Загрузка модели

# %%
model = YOLO(MODEL_PATH)
model.fuse()
class_names = model.names
num_classes = len(class_names)
print(f"Модель: {num_classes} классов")

CLASS_COLORS = {}
for cid in class_names:
    hue = int(cid * 180 / max(num_classes, 1)) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 240]]]), cv2.COLOR_HSV2BGR)[0][0]
    CLASS_COLORS[cid] = tuple(int(c) for c in bgr)

# %% [markdown]
# ## Утилиты

# %%
def compute_resize(w, h):
    s = math.sqrt(TARGET_AREA / (w * h))
    return max(32, int(round(w * s / 32) * 32)), max(32, int(round(h * s / 32) * 32))

def is_touching_border(bbox, fw, fh, margin=None, sides=None):
    if margin is None:
        margin = max(BORDER_MARGIN_PX, int(min(fw, fh) * BORDER_MARGIN_PERCENT))
    if sides is None:
        sides = BORDER_SIDES
    x1, y1, x2, y2 = bbox
    touched = set()
    if x1 <= margin: touched.add("left")
    if x2 >= fw - margin: touched.add("right")
    if y2 >= fh - margin: touched.add("bottom")
    if y1 <= margin: touched.add("top")
    return bool(touched & sides)

def format_picket(raw_position):
    """Преобразует position из SFF в читаемый формат км+метры.
    SFF хранит position как целое число: первые цифры = км, остальные = метры.
    Например: 5300 -> 'км 5+300', 12050 -> 'км 12+050'."""
    pos = int(raw_position)
    s = str(abs(pos))
    if len(s) <= 3:
        val = f"км 0+{s.zfill(3)}"
    else:
        km = s[:-3]
        m = s[-3:]
        val = f"км {km}+{m}"
    return f"-{val}" if pos < 0 else val

def format_picket_raw(raw_position):
    """Возвращает сырое числовое значение пикета."""
    return int(raw_position)

def get_road_position(x1, x2, img_w):
    r = (x1 + x2) / 2 / img_w
    if r < 0.33: return "Левая полоса"
    elif r < 0.66: return "Центр дороги"
    return "Правая полоса"

def preview_border(sff_path, frame_number=0):
    """Визуализация зоны границы для отладки BORDER_MARGIN_PX."""
    reader = SFFReader(sff_path)
    frame = reader.get_frame_by_number(frame_number)
    h, w = frame.shape[:2]
    vis = frame.copy()
    y = h - BORDER_MARGIN_PX
    cv2.line(vis, (0, y), (w, y), (0, 0, 255), 2)
    ov = vis.copy()
    cv2.rectangle(ov, (0, y), (w, h), (0, 0, 255), -1)
    vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Border zone -- {os.path.basename(sff_path)}, frame {frame_number}")
    plt.show()

def draw_info_bar(img, frame_idx, picket_str, num_det, ts_str=""):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, picket_str, (15, 35), f, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if ts_str:
        cv2.putText(img, ts_str, (w // 3, 35), f, 0.7, (180, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(img, f"F:{frame_idx}", (2*w//3, 35), f, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    c = (0, 255, 0) if num_det > 0 else (100, 100, 100)
    txt = f"Obj: {num_det}"
    (tw, _), _ = cv2.getTextSize(txt, f, 0.8, 2)
    cv2.putText(img, txt, (w - tw - 15, 35), f, 0.8, c, 2, cv2.LINE_AA)

def annotate_frame(frame, boxes, tids, cids, exited=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    lc = (0, 255, 0) if exited else (0, 0, 255)
    cv2.line(vis, (0, h - BORDER_MARGIN_PX), (w, h - BORDER_MARGIN_PX), lc, 2)
    for box, tid, cid in zip(boxes, tids, cids):
        x1, y1, x2, y2 = map(int, box)
        near = is_touching_border((x1, y1, x2, y2), w, h)
        color = (0, 165, 255) if near else CLASS_COLORS.get(int(cid), (255, 140, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        lbl = f"{class_names.get(int(cid), cid)} #{tid}"
        f = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(lbl, f, 0.5, 1)
        ov = vis.copy()
        cv2.rectangle(ov, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.addWeighted(ov, 0.75, vis, 0.25, 0, vis)
        cv2.putText(vis, lbl, (x1 + 2, y1 - 4), f, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

# %% [markdown]
# ## BorderExitTracker

# %%
class BorderExitTracker:
    def __init__(self):
        self.active = {}
        self.logged = set()

    def reset(self):
        self.active.clear()
        self.logged.clear()

    def update_frame(self, detections, frame_idx, frame_info, fw, fh):
        current = {tid for tid, _, _ in detections}
        events = []
        for tid in set(self.active) - current:
            st = self.active.pop(tid)
            if tid in self.logged:
                continue
            touching = is_touching_border(st["bbox"], st["fw"], st["fh"])
            if DEBUG_BORDER_EXITS and not touching:
                gap = st["fh"] - st["bbox"][3]
                margin = max(BORDER_MARGIN_PX, int(st["fh"] * BORDER_MARGIN_PERCENT))
                print(f"[debug] track {tid} lost, gap={gap:.0f}px margin={margin}")
            if touching:
                events.append({"track_id": int(tid), "class_id": int(st["class_id"]),
                               "bbox": st["bbox"],
                               "frame_idx": st["frame_idx"], "frame_info": st["frame_info"]})
                self.logged.add(tid)
        for tid, bbox, cid in detections:
            self.active[tid] = {"bbox": bbox, "frame_idx": frame_idx,
                                "frame_info": frame_info, "class_id": cid, "fw": fw, "fh": fh}
        return events

    def flush(self):
        events = []
        for tid, st in list(self.active.items()):
            if tid in self.logged:
                continue
            # Если видео кончилось, трек обрывается - отдаем его как есть
            events.append({"track_id": int(tid), "class_id": int(st["class_id"]),
                           "bbox": st["bbox"],
                           "frame_idx": st["frame_idx"], "frame_info": st["frame_info"]})
            self.logged.add(tid)
        return events

def reset_tracker_state(mdl):
    try:
        if getattr(mdl, "predictor", None):
            for t in getattr(mdl.predictor, "trackers", []):
                t.reset()
    except Exception:
        pass

# %% [markdown]
# ## Batch-обработка (только CSV)

# %%
def process_video_batch(mdl, sff_path):
    reader = SFFReader(sff_path)
    if not reader.frame_data:
        print(f"[!] Skip {sff_path}: no .dat")
        return [], {}

    reset_tracker_state(mdl)
    bt = BorderExitTracker()
    events = []
    road_name = reader.header.get("road_name", "unknown")
    start_km = reader.header.get("start_km", 0)
    end_km = reader.header.get("end_km", 0)
    direction = reader.header.get("direction", 0)
    meta = {"road_name": road_name, "start_km": start_km, "end_km": end_km,
            "direction": direction, "sff_path": sff_path, "total_frames": len(reader.frame_data)}

    for fi, finfo in enumerate(tqdm(reader.frame_data, desc=os.path.basename(sff_path), leave=False)):
        frame = reader._read_frame(finfo["offset"], finfo["jpeg_size"])
        if frame is None:
            continue
        h, w = frame.shape[:2]
        nw, nh = compute_resize(w, h)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        results = mdl.track(resized, persist=True, conf=CONF_THRESHOLD,
                            classes=CLASSES_FILTER, tracker=TRACKER_CFG, verbose=False)
        r = results[0]
        boxes, tids, cids = [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.int().cpu().tolist()
            cids = r.boxes.cls.int().cpu().tolist()

        # Фильтр по минимальной площади
        dets = [(tid, box, cid) for tid, box, cid in zip(tids, boxes, cids)
                if (box[2]-box[0]) * (box[3]-box[1]) >= MIN_BOX_AREA]

        for ev in bt.update_frame(dets, fi, finfo, nw, nh):
            pos = ev["frame_info"]["position"]
            bbox = ev["bbox"]
            events.append({
                "source_file": os.path.basename(sff_path),
                "road_name": road_name,
                "track_id": ev["track_id"],
                "class_id": ev["class_id"],
                "class_name": mdl.names.get(ev["class_id"], str(ev["class_id"])),
                "picket": format_picket(pos),
                "picket_raw": format_picket_raw(pos),
                "road_position": get_road_position(bbox[0], bbox[2], nw),
                "frame_number": ev["frame_idx"],
                "timestamp": str(ev["frame_info"].get("timestamp", "")),
            })

    for ev in bt.flush():
        pos = ev["frame_info"]["position"]
        bbox = ev["bbox"]
        events.append({
            "source_file": os.path.basename(sff_path),
            "road_name": road_name,
            "track_id": ev["track_id"],
            "class_id": ev["class_id"],
            "class_name": mdl.names.get(ev["class_id"], str(ev["class_id"])),
            "picket": format_picket(pos),
            "picket_raw": format_picket_raw(pos),
            "road_position": get_road_position(bbox[0], bbox[2], nw),
            "frame_number": ev["frame_idx"],
            "timestamp": str(ev["frame_info"].get("timestamp", "")),
        })
    return events, meta

def merge_nearby_events(events, radius=MERGE_PICKET_RADIUS):
    """Объединяет события одного класса ближе radius пикетов (с учетом стороны дороги)."""
    if not events:
        return events
    by_class = defaultdict(list)
    for ev in events:
        # Группируем по файлу, классу и стороне дороги!
        by_class[(ev["source_file"], ev["class_name"], ev["road_position"])].append(ev)

    merged = []
    for key, group in by_class.items():
        group.sort(key=lambda e: e["picket_raw"])
        current = group[0].copy()
        for ev in group[1:]:
            if ev["picket_raw"] - current["picket_raw"] <= radius:
                continue  # дубль, пропускаем
            else:
                merged.append(current)
                current = ev.copy()
        merged.append(current)
    merged.sort(key=lambda e: e["picket_raw"])
    return merged

# %% [markdown]
# ## Demo-обработка (видео + снимки)

# %%
def run_visual_demo(sff_path, mdl, output_dir, start_frame=0, num_frames=None):
    reader = SFFReader(sff_path)
    if not reader.frame_data:
        raise ValueError(f"No .dat for {sff_path}")

    reset_tracker_state(mdl)
    bt = BorderExitTracker()
    events = []
    total = len(reader.frame_data)
    end = min(start_frame + num_frames, total) if num_frames else total

    sample = reader.get_frame_by_number(start_frame)
    oh, ow = sample.shape[:2]
    nw, nh = compute_resize(ow, oh)

    vpath = str(output_dir / "demo_annotated.mp4")
    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (nw, nh))
    snap_dir = output_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    last_snap = {}

    for fi in tqdm(range(start_frame, end), desc="Demo"):
        finfo = reader.frame_data[fi]
        frame = reader._read_frame(finfo["offset"], finfo["jpeg_size"])
        if frame is None:
            continue
        pos = finfo["position"]
        ts = finfo.get("timestamp", None)
        ts_str = ts.strftime("%H:%M:%S") if ts else ""
        picket_str = format_picket(pos)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        results = mdl.track(resized, persist=True, conf=CONF_THRESHOLD,
                            classes=CLASSES_FILTER, tracker=TRACKER_CFG, verbose=False)
        r = results[0]
        boxes, tids, cids = [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.int().cpu().tolist()
            cids = r.boxes.cls.int().cpu().tolist()

        dets = [(tid, box, cid) for tid, box, cid in zip(tids, boxes, cids)
                if (box[2]-box[0]) * (box[3]-box[1]) >= MIN_BOX_AREA]

        fevents = bt.update_frame(dets, fi, finfo, nw, nh)
        exited = {ev["track_id"] for ev in fevents}
        for ev in fevents:
            events.append({"track_id": ev["track_id"],
                           "class_name": mdl.names.get(ev["class_id"], str(ev["class_id"])),
                           "picket": picket_str, "frame": ev["frame_idx"]})

        vis = annotate_frame(resized, [d[1] for d in dets], [d[0] for d in dets],
                             [d[2] for d in dets], exited)
        draw_info_bar(vis, fi, picket_str, len(dets), ts_str)
        writer.write(vis)

        # Снимки и кропы
        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy()
            for bi in range(len(confs)):
                if confs[bi] >= CONF_SNAPSHOT_MIN and bi < len(cids):
                    cid = cids[bi]
                    lp = last_snap.get(cid, -9999)
                    if abs(pos - lp) >= SNAPSHOT_PICKET_GAP:
                        last_snap[cid] = pos
                        # Снимок кадра
                        cd = snap_dir / class_names.get(cid, "unknown")
                        cd.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(cd / f"p{pos}_f{fi}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        # Кроп объекта
                        if bi < len(boxes):
                            x1, y1, x2, y2 = map(int, boxes[bi])
                            crop = resized[max(0,y1):y2, max(0,x1):x2]
                            if crop.size > 0:
                                crd = crop_dir / class_names.get(cid, "unknown")
                                crd.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(crd / f"crop_p{pos}_f{fi}.jpg"), crop)

    for ev in bt.flush():
        events.append({"track_id": ev["track_id"],
                       "class_name": mdl.names.get(ev["class_id"], str(ev["class_id"])),
                       "picket": format_picket(ev["frame_info"]["position"]),
                       "frame": ev["frame_idx"]})

    writer.release()
    print(f"Video: {vpath}, Snaps: {len(list(snap_dir.rglob('*.jpg')))}, "
          f"Crops: {len(list(crop_dir.rglob('*.jpg')))}")
    return events

# %% [markdown]
# ## Запуск batch

# %%
sff_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, SFF_GLOB_PATTERN), recursive=True))
print(f"Найдено {len(sff_files)} .sff файлов")

all_events = []
all_meta = []
for sff_path in tqdm(sff_files, desc="Видео"):
    evts, meta = process_video_batch(model, sff_path)
    all_events.extend(evts)
    all_meta.append(meta)

# Мерж дублей
all_events = merge_nearby_events(all_events)
print(f"Уникальных объектов (после мержа): {len(all_events)}")

# %% [markdown]
# ## Итоговый отчет

# %%
if all_events:
    df = pd.DataFrame(all_events)
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = BASE_OUTPUT_DIR / f"detections_{_run_ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSON метаданные запуска
    run_meta = {
        "timestamp": _run_ts,
        "model": MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "tracker": TRACKER_CFG,
        "min_box_area": MIN_BOX_AREA,
        "merge_picket_radius": MERGE_PICKET_RADIUS,
        "border_margin_px": BORDER_MARGIN_PX,
        "videos_processed": len(sff_files),
        "total_events": len(all_events),
        "video_meta": all_meta,
    }
    with open(BASE_OUTPUT_DIR / f"run_meta_{_run_ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 70)
    print(f"Объектов: {len(df)}")
    for cn, grp in df.groupby("class_name"):
        print(f"  {cn:<25s}  {len(grp):3d} шт.")
    print(f"\nCSV: {csv_path}")
    print("\nОбъекты:")
    for _, r in df.iterrows():
        print(f"  {r['class_name']:<20s} | {r['picket']} | {r['road_name']} | Track #{r['track_id']}")
else:
    print("Объектов не найдено.")

# %% [markdown]
# ## (Опционально) Demo-видео

# %%
DEMO_SFF = sff_files[0] if sff_files else None
if DEMO_SFF:
    demo_dir = BASE_OUTPUT_DIR / f"demo_{_run_ts}"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_events = run_visual_demo(DEMO_SFF, model, demo_dir, 0)
    for e in demo_events: print(e)

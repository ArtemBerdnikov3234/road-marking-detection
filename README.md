# Road Marking Detection

Automated detection, classification, and geospatial cataloging of horizontal road markings on video from mobile road laboratories using YOLOv8 and ByteTrack.

## Overview

This system processes video recordings in proprietary SFF (Sensor Fusion Format) from mobile road diagnostic laboratories. It detects 26 classes of horizontal road markings per GOST R 51256-2018, tracks objects across frames, deduplicates detections by road chainage (picket), and generates structured CSV reports.

**Key features:**
- YOLOv8 Instance Segmentation / YOLO26s Detection for 26 marking classes
- ByteTrack multi-object tracking with border-exit detection algorithm
- Automated dataset pipeline: EDA → split → validation → filtering → oversampling → packaging
- Native SFF video format support (frames + telemetry: GPS, chainage, timestamps)
- CSV reports with road chainage binding and annotated demo video generation

## Project Structure

```
├── 04_notebooks/              # Training & inference notebooks
│   ├── 01_train_yolov8_seg.py    # YOLOv8n-seg local training (DirectML/CPU)
│   ├── 02_train_model.py         # Training with visualization & analysis
│   ├── 03_train_colab.py         # YOLO26s training on Google Colab (T4 GPU)
│   └── 04_sff_inference.py       # SFF video inference pipeline
│
├── 05_scripts/                # Data processing pipeline
│   ├── config.yaml               # Pipeline configuration
│   ├── run_pipeline.py           # Master pipeline runner
│   ├── 01_eda_raw.py             # Exploratory data analysis
│   ├── 02_split_dataset.py       # Train/val/test split (no data leakage)
│   ├── 03_validate_dataset.py    # Annotation validation
│   ├── 04_filter_horizontal_only.py  # Filter horizontal markings (26 of 37 classes)
│   ├── 05_oversample_rare_classes.py # Class balancing via oversampling
│   ├── 06_zip_dataset.py         # Dataset packaging
│   ├── 06_predict_all_images.py  # Batch prediction on images
│   ├── 07_model_report.py        # Model metrics & charts
│   └── 08_upload_dataset.py      # Google Drive upload
│
├── 01_raw_data/               # Raw SFF videos & CVAT exports (not in repo)
├── 02_yolo_dataset/           # Processed YOLO dataset (not in repo)
├── 03_models/                 # Trained model weights (not in repo)
├── 06_inference_results/      # Inference outputs (not in repo)
└── 06_reports/                # Generated reports (not in repo)
```

## Pipeline

### Data Preparation
```bash
cd 05_scripts
python run_pipeline.py --raw_data <path_to_raw_data>
```

This sequentially runs:
1. **EDA** — validates category consistency across CVAT tasks
2. **Split** — groups frames by video segment to prevent data leakage, splits 70/15/15
3. **Validate** — checks label format, class IDs, cross-split leakage
4. **Filter** — removes longitudinal markings (classes 1.1–1.11), remaps IDs, rebalances splits
5. **Oversample** — balances rare classes with color augmentation (Albumentations)
6. **Package** — creates zip archive for upload

### Training
- **Local (CPU/DirectML):** `04_notebooks/01_train_yolov8_seg.py` — YOLOv8n-seg
- **Google Colab (T4 GPU):** `04_notebooks/03_train_colab.py` — YOLO26s-det

### Inference
`04_notebooks/04_sff_inference.py` — processes SFF videos:
- Reads frames and telemetry via SFFReader
- Runs YOLO detection + ByteTrack tracking
- BorderExitTracker captures objects exiting frame borders
- Deduplicates by chainage (merge radius: 30m)
- Outputs: CSV report + annotated demo video + object snapshots/crops

## Marking Classes (26)

Horizontal road markings per GOST R 51256-2018:

| ID | Marking | ID | Marking |
|---|---|---|---|
| 0 | 1.12 | 13 | 1.20 |
| 1 | 1.13 | 14 | 1.22 |
| 2 | 1.14.1 | 15 | 1.23.1 |
| 3 | 1.14.2 | 16 | 1.23.2 |
| 4 | 1.14.3 | 17 | 1.23.3 |
| 5 | 1.15 | 18 | 1.24.1 |
| 6 | 1.16.1 | 19 | 1.24.2 |
| 7 | 1.16.2 | 20 | 1.24.3 |
| 8 | 1.16.3 | 21 | 1.24.4 |
| 9 | 1.17.1 | 22 | 1.25 |
| 10 | 1.17.2 | 23 | 1.26 |
| 11 | 1.18 | 24 | ШП |
| 12 | 1.19 | 25 | 1.21 |

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended for training)
- Ultralytics YOLOv8
- OpenCV, Pandas, NumPy
- Albumentations (for oversampling)

```bash
pip install -r 05_scripts/requirements.txt
```

## License

This project was developed as part of an internship at a road diagnostics organization.

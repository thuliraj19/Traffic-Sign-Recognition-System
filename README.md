# Traffic Sign Recognition System for Autonomous Vehicles

A two-stage deep learning pipeline for real-time traffic sign detection (YOLOv8)
and classification (CNN / MobileNetV2 / EfficientNet-B0), optimized for edge
deployment on Raspberry Pi / Jetson Nano via ONNX + quantization using the
**Mapillary Traffic Sign Dataset (MTSD)**.

## Project Structure

```
traffic_sign_project/
├── data/
│   ├── prepare_mapillary_yolo.py           # Prepares Mapillary full scenes for YOLOv8
│   ├── prepare_mapillary_classification.py # Prepares cropped sign images for classification
│   ├── sample_mapillary_subset.py          # Creates balanced training/val subsets
│   └── augmentations.py                    # Weather / blur / occlusion / lighting augmentations
├── scripts/
│   ├── train_classifier.py                 # Trains baseline CNN, MobileNetV2, EfficientNet-B0
│   ├── train_detector.py                   # Trains YOLOv8 on Mapillary
│   ├── pipeline_inference.py               # End-to-end detect -> crop -> classify
│   ├── optimize_model.py                   # ONNX conversion + INT8 quantization
│   ├── benchmark_edge.py                   # FPS / latency benchmarking (Pi / Jetson / CPU)
│   └── evaluate.py                         # mAP, IoU, Accuracy, F1, confusion matrix
├── models/                                 # Saved weights (.pt, .onnx) land here
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare Mapillary dataset (YOLO detection + Classification crops)
python data/prepare_mapillary_yolo.py --raw_dir data/Mapillary_raw --split val
python data/sample_mapillary_subset.py --source_dir data/mapillary_yolo --output_dir data/mapillary_yolo_subset
python data/prepare_mapillary_classification.py --raw_dir data/Mapillary_raw --split val --output_dir data/mapillary_classification

# 3. Train the detector (YOLOv8)
python scripts/train_detector.py --data data/mapillary_yolo_subset/data.yaml --epochs 100

# 4. Train the classifier (choose backbone)
python scripts/train_classifier.py --data_dir data/mapillary_classification --backbone efficientnet_b0 --epochs 30

# 5. Run the full end-to-end pipeline on a video or image
python scripts/pipeline_inference.py --source path/to/video.mp4 \
    --detector models/detector_runs/yolov8n_traffic_sign/weights/best.pt \
    --classifier models/efficientnet_b0_best.pt

# 6. Optimize for edge deployment
python scripts/optimize_model.py --model models/efficientnet_b0_best.pt \
    --output models/efficientnet_b0_quantized.onnx --quantize

# 7. Benchmark on target hardware
python scripts/benchmark_edge.py --detector models/detector_runs/yolov8n_traffic_sign/weights/best.onnx \
    --classifier models/efficientnet_b0_quantized.onnx
```

## Dataset
- **Mapillary Traffic Sign Dataset (MTSD)** — Diverse, global street-level imagery covering varied weather, lighting, viewpoints, and resolutions. Grouped into 4 core functional sign categories:
  1. `danger` (warning & hazard signs)
  2. `mandatory` (directional & compulsory instructions)
  3. `prohibitory` (speed limits, restrictions, and no-entry signs)
  4. `other` (priority, yield, stop, and information signs)

## Pipeline Overview
`Input frame -> YOLOv8 detector -> bounding boxes -> crop regions ->
CNN/MobileNetV2/EfficientNet-B0 classifier -> category label + confidence`


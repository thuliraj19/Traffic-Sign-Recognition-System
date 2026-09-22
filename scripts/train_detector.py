"""
train_detector.py
------------------
Trains a YOLOv8 model on the Mapillary dataset (prepared via data/prepare_mapillary_yolo.py
or data/sample_mapillary_subset.py) to localize traffic signs in full driving-scene images.

Usage:
    python train_detector.py --data data/mapillary_yolo_subset/data.yaml --epochs 100 --model_size n
"""

import argparse
from ultralytics import YOLO


def main(args):
    # model_size: n (nano) / s (small) / m (medium) -- nano/small recommended
    # for edge deployment (Raspberry Pi / Jetson Nano) due to lower latency.
    model = YOLO(f"yolov8{args.model_size}.pt")  # loads pretrained COCO weights

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=args.device,
        project=args.output_dir,
        name=f"yolov8{args.model_size}_traffic_sign",
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # illumination/color augmentation
        degrees=10.0,        # rotation, simulates viewpoint changes
        translate=0.1,
        scale=0.5,           # simulates varying sign distance
        fliplr=0.0,          # DO NOT flip traffic signs horizontally (changes meaning)
        flipud=0.0,
        val=True,
        plots=True,
    )

    print("\nTraining complete.")
    print(f"Best weights saved at: {args.output_dir}/yolov8{args.model_size}_traffic_sign/weights/best.pt")

    # Run final validation to report mAP/IoU
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/mapillary_yolo_subset/data.yaml",
                         help="Path to data.yaml")
    parser.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l"],
                         help="YOLOv8 variant: n=nano (fastest, best for edge), s=small, m=medium, l=large")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--output_dir", type=str, default="models/detector_runs")
    args = parser.parse_args()

    main(args)

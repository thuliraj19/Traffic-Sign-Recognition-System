"""
pipeline_inference.py
----------------------
End-to-end detect -> crop -> classify pipeline.

Takes an image, video, or webcam stream, runs the YOLOv8 detector to locate
traffic signs, crops each detected region, and passes it through the trained
classifier (baseline CNN / MobileNetV2 / EfficientNet-B0) to predict the
exact sign class. Draws results and reports per-frame FPS.

Usage:
    python pipeline_inference.py --source path/to/video.mp4 \
        --detector models/yolov8n_traffic_sign/weights/best.pt \
        --classifier models/efficientnet_b0_best.pt
"""

import argparse
import time
import sys
import os

import cv2
import torch
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_classifier import build_model
from data.augmentations import get_val_transforms

# Mapillary 4-category label names (ImageFolder alphabetical default)
MAPILLARY_CLASS_NAMES = ["danger", "mandatory", "other", "prohibitory"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficSignPipeline:
    def __init__(self, detector_path, classifier_path, conf_threshold=0.4):
        print("Loading YOLOv8 detector...")
        self.detector = YOLO(detector_path)
        self.conf_threshold = conf_threshold

        print("Loading classifier...")
        checkpoint = torch.load(classifier_path, map_location=DEVICE)
        self.classifier = build_model(checkpoint["backbone"], checkpoint["num_classes"])
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.to(DEVICE).eval()
        self.img_size = checkpoint.get("img_size", 64)
        self.class_names = checkpoint.get("classes", MAPILLARY_CLASS_NAMES)
        self.transform = get_val_transforms(img_size=self.img_size)

        print(f"Pipeline ready. Detector: {detector_path} | Classifier: {classifier_path} "
              f"({checkpoint['backbone']}, val_f1={checkpoint.get('val_f1', 'n/a')})")

    @torch.no_grad()
    def classify_crop(self, crop_bgr):
        """Runs the classifier on a single cropped sign region (BGR numpy array)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=crop_rgb)["image"].unsqueeze(0).to(DEVICE)
        logits = self.classifier(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        idx = pred_idx.item()
        class_name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
        return class_name, conf.item()

    def process_frame(self, frame):
        """Runs detection + classification on a single frame. Returns annotated frame."""
        t0 = time.time()
        detections = self.detector(frame, conf=self.conf_threshold, verbose=False)[0]

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            # Guard against degenerate boxes
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            class_name, cls_conf = self.classify_crop(crop)

            label = f"{class_name} ({cls_conf*100:.0f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame, fps

    def run_on_image(self, image_path, output_path="output.jpg"):
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read input image: {image_path}")
        annotated, fps = self.process_frame(frame)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        success = cv2.imwrite(output_path, annotated)
        if not success:
            raise IOError(f"Failed to save annotated image to {output_path}")
        print(f"Saved annotated image to {output_path} (FPS={fps:.1f})")

    def run_on_video(self, source, output_path="output.mp4", show=False):
        cap = cv2.VideoCapture(source)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        fps_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, fps = self.process_frame(frame)
            fps_log.append(fps)
            writer.write(annotated)

            if show:
                cv2.imshow("Traffic Sign Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        if fps_log:
            print(f"Saved annotated video to {output_path}")
            print(f"Average FPS: {np.mean(fps_log):.2f} | Min: {np.min(fps_log):.2f} | Max: {np.max(fps_log):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                         help="Path to image/video, or '0' for webcam")
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    pipeline = TrafficSignPipeline(args.detector, args.classifier, args.conf)

    ext = os.path.splitext(args.source)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        pipeline.run_on_image(args.source, f"{args.output}.jpg")
    else:
        source = int(args.source) if args.source == "0" else args.source
        pipeline.run_on_video(source, f"{args.output}.mp4", show=args.show)


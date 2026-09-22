"""
evaluate.py
------------
Evaluation utilities for both stages:
  - Detection: mAP / IoU (via Ultralytics' built-in validator)
  - Classification: Accuracy, F1, confusion matrix, per-class breakdown
  - Quantization-aware error analysis: compares FP32 vs INT8 accuracy,
    broken down per class, to identify which sign types degrade most
    after quantization.

Usage:
    python evaluate.py --mode classifier --model models/efficientnet_b0_best.pt \
        --data_dir data/mapillary_classification/val

    python evaluate.py --mode quant_compare --fp32_onnx models/efficientnet_b0.onnx \
        --int8_onnx models/efficientnet_b0_int8.onnx --data_dir data/mapillary_classification/val
"""

import argparse
import sys
import os

import torch
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_classifier import build_model, AlbumentationsImageFolder
from data.augmentations import get_val_transforms
from scripts.pipeline_inference import MAPILLARY_CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_classifier_pt(model_path, data_dir, img_size=64, batch_size=64):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = build_model(checkpoint["backbone"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    class_names = checkpoint.get("classes", MAPILLARY_CLASS_NAMES)
    dataset = AlbumentationsImageFolder(data_dir, get_val_transforms(img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Backbone: {checkpoint['backbone']}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro F1: {f1:.4f}\n")

    unique_labels = sorted(list(set(all_labels)))
    target_names = [class_names[i] if i < len(class_names) else str(i) for i in unique_labels]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    plot_confusion_matrix(all_labels, all_preds, labels=target_names,
                          save_path=f"confusion_matrix_{checkpoint['backbone']}.png")
    return acc, f1


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels if labels else "auto",
                yticklabels=labels if labels else "auto")
    plt.title("Confusion Matrix - Traffic Sign Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved: {save_path}")


def run_onnx_inference(session, input_name, image_batch):
    outputs = session.run(None, {input_name: image_batch})
    return np.argmax(outputs[0], axis=1)


def compare_fp32_vs_int8(fp32_onnx_path, int8_onnx_path, data_dir, img_size=64, batch_size=32):
    """
    Quantization-aware error analysis: runs both FP32 and INT8 ONNX models
    on the test/val set and reports per-class accuracy drop.
    """
    fp32_session = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
    fp32_input = fp32_session.get_inputs()[0].name
    int8_input = int8_session.get_inputs()[0].name

    dataset = AlbumentationsImageFolder(data_dir, get_val_transforms(img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fp32_preds, int8_preds, all_labels = [], [], []

    for images, labels in loader:
        images_np = images.numpy().astype(np.float32)
        fp32_preds.extend(run_onnx_inference(fp32_session, fp32_input, images_np))
        int8_preds.extend(run_onnx_inference(int8_session, int8_input, images_np))
        all_labels.extend(labels.numpy())

    fp32_acc = accuracy_score(all_labels, fp32_preds)
    int8_acc = accuracy_score(all_labels, int8_preds)

    print(f"FP32 Accuracy: {fp32_acc:.4f}")
    print(f"INT8 Accuracy: {int8_acc:.4f}")
    print(f"Accuracy drop: {(fp32_acc - int8_acc) * 100:.2f} percentage points\n")

    # Per-class breakdown
    num_classes = len(set(all_labels))
    print(f"{'Class':<20} {'FP32 Acc':>10} {'INT8 Acc':>10} {'Drop':>8}")
    print("-" * 52)
    per_class_drop = []
    for cls in range(num_classes):
        idx = [i for i, l in enumerate(all_labels) if l == cls]
        if not idx:
            continue
        cls_labels = [all_labels[i] for i in idx]
        cls_fp32 = [fp32_preds[i] for i in idx]
        cls_int8 = [int8_preds[i] for i in idx]

        fp32_cls_acc = accuracy_score(cls_labels, cls_fp32)
        int8_cls_acc = accuracy_score(cls_labels, cls_int8)
        drop = fp32_cls_acc - int8_cls_acc
        name = MAPILLARY_CLASS_NAMES[cls] if cls < len(MAPILLARY_CLASS_NAMES) else str(cls)
        per_class_drop.append((name, drop))

        print(f"{name:<20} {fp32_cls_acc:>10.3f} {int8_cls_acc:>10.3f} {drop:>8.3f}")

    # Highlight most-affected classes
    per_class_drop.sort(key=lambda x: -x[1])
    print("\nClasses affected by quantization:")
    for name, drop in per_class_drop:
        print(f"  {name}: {drop*100:.2f} pp accuracy drop")

    return fp32_acc, int8_acc, per_class_drop


def evaluate_detector(model_path, data_yaml):
    """Wraps Ultralytics' built-in validator to report mAP50, mAP50-95, and IoU stats."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                         choices=["classifier", "detector", "quant_compare"])
    parser.add_argument("--model", type=str, help="Path to .pt classifier or detector")
    parser.add_argument("--data_dir", type=str, default="data/mapillary_classification/val",
                         help="Path to validation data directory")
    parser.add_argument("--data_yaml", type=str, default="data/mapillary_yolo_subset/data.yaml",
                         help="Path to data.yaml (detector mode)")
    parser.add_argument("--fp32_onnx", type=str, help="Path to FP32 ONNX model")
    parser.add_argument("--int8_onnx", type=str, help="Path to INT8 quantized ONNX model")
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    if args.mode == "classifier":
        evaluate_classifier_pt(args.model, args.data_dir, args.img_size)
    elif args.mode == "detector":
        evaluate_detector(args.model, args.data_yaml)
    elif args.mode == "quant_compare":
        compare_fp32_vs_int8(args.fp32_onnx, args.int8_onnx, args.data_dir, args.img_size)

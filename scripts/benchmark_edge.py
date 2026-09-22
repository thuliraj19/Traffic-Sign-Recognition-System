"""
benchmark_edge.py
-------------------
Benchmarks the full detect+classify pipeline's real-world inference speed
on the current device (intended to be run directly on Raspberry Pi / Jetson
Nano / laptop CPU to get honest, non-GPU-inflated FPS numbers).

This directly addresses the gap flagged in the literature review: most
papers report GPU-only benchmarks, not actual edge-hardware throughput
for the COMBINED pipeline (detector + classifier back to back).

Usage:
    python benchmark_edge.py --detector models/yolov8n.onnx \
        --classifier models/efficientnet_b0_int8.onnx --n_frames 200
"""

import argparse
import time
import platform

import cv2
import numpy as np
import onnxruntime as ort


def get_device_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def benchmark_pipeline(detector_path, classifier_path, img_size_det=640,
                        img_size_cls=64, n_frames=200):
    print("Device info:", get_device_info())

    det_session = ort.InferenceSession(detector_path, providers=["CPUExecutionProvider"])
    cls_session = ort.InferenceSession(classifier_path, providers=["CPUExecutionProvider"])

    det_input_name = det_session.get_inputs()[0].name
    cls_input_name = cls_session.get_inputs()[0].name

    # Simulated frame (replace with real camera/video frames for accurate results)
    dummy_frame = np.random.randn(1, 3, img_size_det, img_size_det).astype(np.float32)
    dummy_crop = np.random.randn(1, 3, img_size_cls, img_size_cls).astype(np.float32)

    # Warmup
    for _ in range(5):
        det_session.run(None, {det_input_name: dummy_frame})
        cls_session.run(None, {cls_input_name: dummy_crop})

    detection_times, classification_times, total_times = [], [], []

    for _ in range(n_frames):
        t0 = time.time()
        det_session.run(None, {det_input_name: dummy_frame})
        t1 = time.time()

        # Assume ~2 signs detected per frame on average for benchmarking realism
        for _ in range(2):
            cls_session.run(None, {cls_input_name: dummy_crop})
        t2 = time.time()

        detection_times.append(t1 - t0)
        classification_times.append(t2 - t1)
        total_times.append(t2 - t0)

    def summarize(times, label):
        times_ms = np.array(times) * 1000
        print(f"\n{label}:")
        print(f"  Mean: {times_ms.mean():.2f} ms | Median: {np.median(times_ms):.2f} ms")
        print(f"  Min : {times_ms.min():.2f} ms  | Max: {times_ms.max():.2f} ms")

    summarize(detection_times, "Detection stage latency")
    summarize(classification_times, "Classification stage latency (2 crops)")
    summarize(total_times, "End-to-end pipeline latency")

    avg_total = np.mean(total_times)
    fps = 1.0 / avg_total
    print(f"\n>>> End-to-end pipeline throughput: {fps:.2f} FPS <<<")
    print("Note: Replace dummy inputs with real video frames for production-accurate numbers.")

    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=str, required=True, help="ONNX detector path")
    parser.add_argument("--classifier", type=str, required=True, help="ONNX classifier path")
    parser.add_argument("--img_size_det", type=int, default=640)
    parser.add_argument("--img_size_cls", type=int, default=64)
    parser.add_argument("--n_frames", type=int, default=200)
    args = parser.parse_args()

    benchmark_pipeline(args.detector, args.classifier, args.img_size_det,
                        args.img_size_cls, args.n_frames)

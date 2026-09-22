"""
optimize_model.py
-------------------
Converts a trained PyTorch classifier to ONNX format and optionally applies
INT8 dynamic quantization for edge deployment (Raspberry Pi / Jetson Nano).

Usage:
    # Convert only
    python optimize_model.py --model models/efficientnet_b0_best.pt \
        --output models/efficientnet_b0.onnx

    # Convert + quantize
    python optimize_model.py --model models/efficientnet_b0_best.pt \
        --output models/efficientnet_b0_quantized.onnx --quantize
"""

import argparse
import sys
import os
import time

import torch
import onnx
import onnxruntime as ort
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_classifier import build_model

DEVICE = torch.device("cpu")  # export on CPU for portability


def export_to_onnx(model_path, onnx_output_path, img_size=64, opset=17):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = build_model(checkpoint["backbone"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify the exported model
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export successful and verified: {onnx_output_path}")

    return checkpoint["backbone"], img_size


def quantize_onnx_model(onnx_path, quantized_output_path):
    """Applies dynamic INT8 quantization -- reduces model size ~4x and speeds
    up CPU inference, at a small accuracy cost, which is measured separately
    in evaluate.py's quantization-aware error analysis."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_output_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved: {quantized_output_path}")


def compare_model_sizes(fp32_path, int8_path):
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"\nModel size comparison:")
    print(f"  FP32 ONNX : {fp32_size:.2f} MB")
    print(f"  INT8 ONNX : {int8_size:.2f} MB")
    print(f"  Reduction : {(1 - int8_size / fp32_size) * 100:.1f}%")


def benchmark_onnx_inference(onnx_path, img_size=64, n_runs=100):
    """Quick sanity-check latency benchmark (single-image, CPU)."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    start = time.time()
    for _ in range(n_runs):
        session.run(None, {input_name: dummy_input})
    elapsed = time.time() - start

    avg_latency_ms = (elapsed / n_runs) * 1000
    fps = n_runs / elapsed
    print(f"\nCPU inference benchmark ({n_runs} runs):")
    print(f"  Avg latency: {avg_latency_ms:.2f} ms")
    print(f"  Throughput : {fps:.1f} FPS")
    return avg_latency_ms, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save .onnx model")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--quantize", action="store_true", help="Also apply INT8 quantization")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after export")
    args = parser.parse_args()

    backbone, img_size = export_to_onnx(args.model, args.output, args.img_size)

    if args.quantize:
        quantized_path = args.output.replace(".onnx", "_int8.onnx")
        quantize_onnx_model(args.output, quantized_path)
        compare_model_sizes(args.output, quantized_path)

        if args.benchmark:
            print("\n--- FP32 ONNX ---")
            benchmark_onnx_inference(args.output, img_size)
            print("\n--- INT8 Quantized ONNX ---")
            benchmark_onnx_inference(quantized_path, img_size)
    elif args.benchmark:
        benchmark_onnx_inference(args.output, img_size)



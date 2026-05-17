#!/usr/bin/env python3
"""
Deep Learning Benchmark Script for Qualcomm RB3 Gen2
Compares PyTorch CPU vs ONNX Runtime performance
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Configuration
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50
BATCH_SIZES = [1, 4, 8]
INPUT_SIZE = (3, 224, 224)
NUM_THREADS = 8
ARTIFACT_DIR = Path("artifacts/deployment/hardware_profiling")

def benchmark_pytorch(model_name, batch_sizes=BATCH_SIZES):
    """Benchmark PyTorch model"""
    import torch
    import timm

    torch.set_num_threads(NUM_THREADS)

    print(f"\n[PyTorch] Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=False)
    model.eval()

    results = {}
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *INPUT_SIZE)

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(BENCHMARK_RUNS):
                start = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size * 1000 / avg_time

        results[f"batch_{batch_size}"] = {
            "latency_ms": round(avg_time, 2),
            "std_ms": round(std_time, 2),
            "throughput_fps": round(throughput, 1)
        }
        print(f"  Batch {batch_size}: {avg_time:.2f} ms ({throughput:.1f} fps)")

    return results

def export_onnx(model_name, output_dir=ARTIFACT_DIR / "onnx_models"):
    """Export PyTorch model to ONNX"""
    import torch
    import timm

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f"{model_name}.onnx"

    if output_path.exists():
        print(f"  ONNX model exists: {output_path}")
        return str(output_path)

    print(f"  Exporting {model_name} to ONNX...")
    model = timm.create_model(model_name, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, *INPUT_SIZE)
    torch.onnx.export(
        model, dummy_input, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17
    )
    print(f"  Exported to {output_path}")
    return str(output_path)

def benchmark_onnx(onnx_path, batch_size=1):
    """Benchmark ONNX Runtime model"""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = NUM_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(batch_size, *INPUT_SIZE).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    throughput = batch_size * 1000 / avg_time

    return {
        "latency_ms": round(avg_time, 2),
        "throughput_fps": round(throughput, 1)
    }

def main():
    print("=" * 60)
    print("Deep Learning Benchmark - Qualcomm RB3 Gen2")
    print("=" * 60)

    models = [
        ("mobilenetv2_100", "MobileNetV2"),
        ("resnet18", "ResNet18"),
        ("efficientnet_b0", "EfficientNet-B0")
    ]

    results = {
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
            "input_size": INPUT_SIZE,
            "num_threads": NUM_THREADS
        },
        "pytorch": {},
        "onnx": {}
    }

    # PyTorch benchmarks
    print("\n[1] PyTorch CPU Benchmarks")
    for model_id, model_name in models:
        results["pytorch"][model_name] = benchmark_pytorch(model_id)

    # ONNX export and benchmark
    print("\n[2] ONNX Runtime Benchmarks")
    for model_id, model_name in models[:2]:  # Only first 2 for ONNX
        onnx_path = export_onnx(model_id)
        print(f"\n[ONNX] Benchmarking {model_name}...")
        results["onnx"][model_name] = benchmark_onnx(onnx_path)
        print(f"  Latency: {results['onnx'][model_name]['latency_ms']} ms")

    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = ARTIFACT_DIR / "benchmark_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'PyTorch':<15} {'ONNX':<15} {'Speedup':<10}")
    print("-" * 60)
    for model_name in ["MobileNetV2", "ResNet18"]:
        pt = results["pytorch"][model_name]["batch_1"]["latency_ms"]
        onnx = results["onnx"].get(model_name, {}).get("latency_ms", "N/A")
        if isinstance(onnx, float):
            speedup = f"{pt/onnx:.2f}x"
        else:
            speedup = "N/A"
        print(f"{model_name:<20} {pt:<15} {onnx:<15} {speedup:<10}")

if __name__ == "__main__":
    main()

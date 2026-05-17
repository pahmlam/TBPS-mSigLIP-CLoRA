#!/usr/bin/env python3
"""
SNPE Benchmark Script for Qualcomm RB3 Gen2
Requires DLC models - see instructions below
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path

# Configuration
ARTIFACT_DIR = Path("artifacts/deployment/hardware_profiling")
MODELS_DIR = ARTIFACT_DIR / "dlc_models"
RESULTS_FILE = ARTIFACT_DIR / "snpe_benchmark_results.json"

def check_snpe_tools():
    """Check if SNPE tools are available"""
    tools = {
        "snpe-net-run": False,
        "snpe-throughput-net-run": False,
        "snpe-platform-validator": False
    }
    for tool in tools:
        result = subprocess.run(["which", tool], capture_output=True)
        tools[tool] = result.returncode == 0
    return tools

def check_runtimes():
    """Validate available runtimes"""
    result = subprocess.run(
        ["snpe-platform-validator", "--runtime", "all", "--testRuntime"],
        capture_output=True, text=True
    )
    runtimes = {"cpu": True, "gpu": False, "dsp": False}
    if "SNPE is supported for runtime GPU" in result.stdout:
        runtimes["gpu"] = True
    if "SNPE is supported for runtime DSP" in result.stdout:
        runtimes["dsp"] = True
    return runtimes

def create_input_raw(shape=(1, 3, 224, 224), output_dir=ARTIFACT_DIR / "snpe_inputs"):
    """Create raw input file for SNPE"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data = np.random.randn(*shape).astype(np.float32)
    filepath = Path(output_dir) / "input.raw"
    data.tofile(filepath)

    list_path = Path(output_dir) / "input_list.txt"
    with open(list_path, "w") as f:
        f.write(str(filepath.absolute()) + "\n")

    return filepath, list_path

def benchmark_snpe(dlc_path, runtime="cpu"):
    """Run SNPE benchmark"""
    if not Path(dlc_path).exists():
        return {"error": f"DLC file not found: {dlc_path}"}

    input_file, input_list = create_input_raw()

    runtime_flag = {
        "cpu": "--use_cpu",
        "gpu": "--use_gpu",
        "dsp": "--use_dsp"
    }.get(runtime, "--use_cpu")

    cmd = [
        "snpe-throughput-net-run",
        "--container", str(dlc_path),
        "--input_list", str(input_list),
        runtime_flag,
        "--duration", "10",
        "--perf_profile", "high_performance"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.split("\n"):
            if "Throughput" in line:
                throughput = float(line.split(":")[1].strip().split()[0])
                latency = 1000.0 / throughput
                return {
                    "latency_ms": round(latency, 2),
                    "throughput_fps": round(throughput, 1),
                    "runtime": runtime
                }
        return {"error": "Could not parse throughput"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("SNPE Benchmark for Qualcomm RB3 Gen2")
    print("=" * 60)

    # Check tools
    print("\n[1] Checking SNPE tools...")
    tools = check_snpe_tools()
    for tool, available in tools.items():
        status = "✓" if available else "✗"
        print(f"  {status} {tool}")

    # Check runtimes
    print("\n[2] Checking available runtimes...")
    runtimes = check_runtimes()
    for rt, available in runtimes.items():
        status = "✓" if available else "✗"
        print(f"  {status} {rt.upper()}")

    # Check for DLC models
    print("\n[3] Checking for DLC models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dlc_files = list(MODELS_DIR.glob("*.dlc"))

    if not dlc_files:
        print(f"  ✗ No DLC models found in {MODELS_DIR}/")
        print("\n" + "=" * 60)
        print("HOW TO GET DLC MODELS:")
        print("=" * 60)
        print("""
Option 1: Qualcomm AI Hub (Recommended)
  1. Sign up at https://aihub.qualcomm.com/
  2. Configure: qai-hub configure --api_token YOUR_TOKEN
  3. Convert: qai-hub submit-compile-job --model model.onnx --device "QCS6490"

Option 2: Install Full SNPE SDK
  1. Download from https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk
  2. Convert: snpe-onnx-to-dlc --input_network model.onnx --output_path model.dlc
""")
        return

    print(f"  ✓ Found {len(dlc_files)} DLC model(s)")

    # Run benchmarks
    print("\n[4] Running benchmarks...")
    results = {"models": {}, "runtimes": runtimes, "tools": tools}

    for dlc_file in dlc_files:
        model_name = dlc_file.stem
        results["models"][model_name] = {}

        for runtime, available in runtimes.items():
            if available:
                print(f"  Benchmarking {model_name} on {runtime.upper()}...")
                result = benchmark_snpe(dlc_file, runtime)
                results["models"][model_name][runtime] = result
                if "latency_ms" in result:
                    print(f"    Latency: {result['latency_ms']:.2f} ms")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()

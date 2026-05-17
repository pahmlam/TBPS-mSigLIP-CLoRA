"""
Step 1: Analyze checkpoint — run on any machine with PyTorch.
Reports model size, memory estimates, and compatibility with target device.

Usage:
    python deployment/scripts/analyze_checkpoint.py --ckpt path/to/epoch=53-val_score=51.30.ckpt
"""

import argparse
import os
import sys

import torch

# Add deployment root to path
_deployment_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_deployment_root)
sys.path.insert(0, _deployment_root)
from deploy_utils import TeeLogger

DEFAULT_LOG_DIR = os.path.join(_project_root, "artifacts", "deployment", "logs")


def analyze(ckpt_path: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\n{'='*60}")
    print("CHECKPOINT STRUCTURE")
    print(f"{'='*60}")
    print(f"Top-level keys: {list(ckpt.keys())}")

    state = ckpt.get("state_dict", {})
    print(f"State dict keys: {len(state)}")

    # --- Parameter analysis ---
    total_params = 0
    groups = {
        "vision_encoder": {"params": 0, "bytes": 0},
        "text_encoder": {"params": 0, "bytes": 0},
        "lora_adapters": {"params": 0, "bytes": 0},
        "simclr_mlp": {"params": 0, "bytes": 0},
        "logit_scale_bias": {"params": 0, "bytes": 0},
        "other": {"params": 0, "bytes": 0},
    }

    seen_ptrs = set()  # deduplicate shared parameters (e.g. backbone vs vision_model/text_model)
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue  # skip duplicate reference to same tensor
        seen_ptrs.add(ptr)

        n = v.numel()
        total_params += n
        b = n * v.element_size()

        if "lora" in k.lower():
            g = "lora_adapters"
        elif "vision" in k or "visual" in k:
            g = "vision_encoder"
        elif "text" in k:
            g = "text_encoder"
        elif "simclr" in k:
            g = "simclr_mlp"
        elif "logit" in k:
            g = "logit_scale_bias"
        else:
            g = "other"

        groups[g]["params"] += n
        groups[g]["bytes"] += b

    print(f"\n{'='*60}")
    print("MODEL SIZE BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Module':<25s} {'Params':>12s} {'FP32 (MB)':>10s} {'FP16 (MB)':>10s}")
    print("-" * 60)
    for g, info in sorted(groups.items(), key=lambda x: -x[1]["bytes"]):
        if info["params"] > 0:
            fp32_mb = info["params"] * 4 / 1024**2
            fp16_mb = info["params"] * 2 / 1024**2
            print(f"{g:<25s} {info['params']:>12,} {fp32_mb:>10.1f} {fp16_mb:>10.1f}")

    fp32_total = total_params * 4 / 1024**2
    fp16_total = total_params * 2 / 1024**2
    int8_total = total_params / 1024**2
    print("-" * 60)
    print(f"{'TOTAL':<25s} {total_params:>12,} {fp32_total:>10.1f} {fp16_total:>10.1f}")

    # --- Memory estimates for inference ---
    # Activations estimate: ~1.5x model size for single sample forward pass
    activation_overhead = 0.5  # conservative multiplier

    print(f"\n{'='*60}")
    print("MEMORY ESTIMATES FOR INFERENCE (single sample)")
    print(f"{'='*60}")
    for dtype, size_mb in [("FP32", fp32_total), ("FP16", fp16_total), ("INT8", int8_total)]:
        total_est = size_mb * (1 + activation_overhead)
        print(f"{dtype}: model={size_mb:.0f}MB + activations~{size_mb*activation_overhead:.0f}MB = ~{total_est:.0f}MB")

    # --- Target device compatibility ---
    target_ram_mb = 4000  # ~4GB available on RB3 Gen2
    print(f"\n{'='*60}")
    print(f"QUALCOMM RB3 GEN2 COMPATIBILITY (available RAM: ~{target_ram_mb}MB)")
    print(f"{'='*60}")
    for dtype, size_mb in [("FP32", fp32_total), ("FP16", fp16_total), ("INT8", int8_total)]:
        total_est = size_mb * (1 + activation_overhead)
        margin = target_ram_mb - total_est
        status = "OK" if margin > 500 else ("TIGHT" if margin > 0 else "NO - OOM risk")
        print(f"{dtype}: ~{total_est:.0f}MB needed, margin={margin:.0f}MB --> {status}")

    # --- LoRA info ---
    lora_keys = [k for k in state.keys() if "lora" in k.lower()]
    print(f"\n{'='*60}")
    print(f"LORA ADAPTERS: {len(lora_keys)} keys")
    print(f"{'='*60}")
    if lora_keys:
        print("First 10:")
        for k in lora_keys[:10]:
            print(f"  {k}: {state[k].shape}")

    # --- Optimizer states ---
    has_optimizer = "optimizer_states" in ckpt
    print(f"\nOptimizer states in checkpoint: {'YES (will be stripped for inference)' if has_optimizer else 'NO'}")

    # --- Recommendations ---
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. Export to FP16 (safest for 4GB RAM): run deployment/scripts/lora_fp16/export.py")
    print("2. Convert to ONNX: run deployment/scripts/onnx/export.py --model-dir artifacts/deployment/exports/msiglip_lora")
    print("3. LoRA weights should be merged into base model before export (step 1 does this)")
    print("4. On RB3: set CPU governor to 'performance' for best speed")
    print("   sudo cpupower frequency-set -g performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    args = parser.parse_args()

    logger = TeeLogger(DEFAULT_LOG_DIR, "analyze")
    analyze(args.ckpt)
    logger.close()

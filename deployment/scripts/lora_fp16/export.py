"""
Step 1: Export checkpoint to FP16 PyTorch format.
Merges LoRA into base model, strips optimizer states, converts to FP16.

Usage:
    python deployment/scripts/lora_fp16/export.py \
        --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
        --output-dir artifacts/deployment/exports/msiglip_lora

Outputs:
    artifacts/deployment/exports/msiglip_lora/
    ├── model_fp16.pt              # PyTorch FP16 state dict (vision + text + logit params)
    ├── model_fp32.pt              # PyTorch FP32 state dict (fallback)
    └── config.yaml                # Resolved Hydra config from checkpoint

Next step:
    python deployment/scripts/onnx/export.py --model-dir artifacts/deployment/exports/msiglip_lora
"""

import argparse
import os
import sys

import torch
import yaml

# Add deployment root to path (lora_fp16/ → scripts/ → deployment/)
_deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _deployment_root)
from deploy_utils import TeeLogger

_project_root = os.path.dirname(_deployment_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

DEFAULT_OUTPUT_DIR = os.path.join(
    _project_root, "artifacts", "deployment", "exports", "msiglip_lora"
)
DEFAULT_LOG_DIR = os.path.join(_project_root, "artifacts", "deployment", "logs")

from omegaconf import OmegaConf


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_model_from_checkpoint(ckpt_path: str):
    """Load Lightning checkpoint and reconstruct model with merged LoRA."""
    from msiglip.lightning_models import LitTBPS

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["config"]
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Build data module just to get train_loader length (needed for LitTBPS init)
    # We use a dummy value since we won't train
    num_iters = 100  # dummy, not used for inference

    # Build model
    lit_model = LitTBPS(config, num_iters_per_epoch=num_iters)

    # Setup LoRA if configured
    if config.get("lora", None):
        lit_model.setup_lora(config.lora)

    # Load state dict
    lit_model.load_state_dict(ckpt["state_dict"], strict=False)
    lit_model.eval()

    # Merge LoRA weights into base model
    try:
        from peft import PeftModel
        if isinstance(lit_model.backbone, PeftModel):
            print("Merging LoRA adapters into base model...")
            lit_model.backbone = lit_model.backbone.merge_and_unload()
            print("LoRA merged successfully.")
    except Exception as e:
        print(f"Warning: Could not merge LoRA: {e}")
        print("Proceeding with unmerged model (LoRA adapters still separate).")

    return lit_model, config


def export_pytorch(lit_model, config, output_dir: str):
    """Export as PyTorch state dicts (FP32 and FP16)."""
    model = lit_model.model

    # Collect inference-relevant state dict
    inference_state = {}
    for k, v in model.state_dict().items():
        inference_state[k] = v

    # FP32
    fp32_path = os.path.join(output_dir, "model_fp32.pt")
    torch.save(inference_state, fp32_path)
    fp32_size = os.path.getsize(fp32_path) / 1024**2
    print(f"Saved FP32: {fp32_path} ({fp32_size:.1f} MB)")

    # FP16 — preserve tensor sharing so torch.save deduplicates properly
    seen = {}  # data_ptr -> converted tensor
    fp16_state = {}
    for k, v in inference_state.items():
        ptr = v.data_ptr()
        if ptr in seen:
            fp16_state[k] = seen[ptr]
        else:
            converted = v.half() if v.is_floating_point() else v
            seen[ptr] = converted
            fp16_state[k] = converted
    fp16_path = os.path.join(output_dir, "model_fp16.pt")
    torch.save(fp16_state, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / 1024**2
    print(f"Saved FP16: {fp16_path} ({fp16_size:.1f} MB)")

    # Save config — use a dumper that represents tuples as plain YAML lists
    config_path = os.path.join(output_dir, "config.yaml")

    class SafeDumper(yaml.SafeDumper):
        pass

    SafeDumper.add_representer(
        tuple,
        lambda dumper, data: dumper.represent_list(data),
    )

    with open(config_path, "w") as f:
        config_dict = OmegaConf.to_container(config, resolve=True)
        yaml.dump(config_dict, f, Dumper=SafeDumper, default_flow_style=False)
    print(f"Saved config: {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lit_model, config = load_model_from_checkpoint(args.ckpt)
    export_pytorch(lit_model, config, args.output_dir)

    print(f"\nDone! All files saved to: {args.output_dir}/")
    print("\nNext step: python deployment/scripts/onnx/export.py --model-dir", args.output_dir)


if __name__ == "__main__":
    logger = TeeLogger(DEFAULT_LOG_DIR, "export_lora_fp16")
    main()
    logger.close()

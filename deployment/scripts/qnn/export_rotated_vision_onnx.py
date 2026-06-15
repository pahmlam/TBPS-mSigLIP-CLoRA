#!/usr/bin/env python3
"""M2 - Export the rotated vision encoder to ONNX opset-20 (fused Gelu).

Plan: `deployment/docs/journal/[deploy]-2026-06-15.md` (sections 5-7).

This mirrors the working `exported_model_gelu20` export (opset 20 so the
tanh-GELU fuses into a single `Gelu` op instead of decomposing into the
`Pow(x,3)` cubic that wrecked quantization) but loads the rotated model via
`load_rotated_model` (NOT `_load_pytorch_model`). The rotation replaced the 25
encoder LayerNorms with paramless RMSNorm; loading with the stock loader rebuilds
mean-subtracting LayerNorm and corrupts the rotated residual (cosine ~0.11).

GATE M2: static-control ONNX-rotated vs PyTorch-rotated cosine ~1.0 on
vn3k_test_10 (run compare_onnx_with_pytorch.py --rotated). Because the rotation
is output-invariant, it should also match the *original* PyTorch model ~1.0.

NOTE: opset 20 has no RMSNormalization op (added in opset 23), so each RMSNorm
exports as a small op cluster (ReduceMean of x*x, Add eps, Sqrt/Reciprocal, Mul).
The script prints op counts so we can see exactly what RMSNorm became and whether
a squared-magnitude intermediate (like the old GELU cubic) needs attention before
quantizing on v68.

Example:
    venv/bin/python deployment/scripts/qnn/export_rotated_vision_onnx.py \\
      --model-dir artifacts/deployment/exports/exported_model_rotated
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import onnx  # noqa: E402

from rotate_vision_encoder import load_rotated_model  # noqa: E402


class VisionWrapper(nn.Module):
    """Wraps TBPS.encode_image for ONNX export (same as scripts/onnx/export.py)."""

    def __init__(self, tbps_model: nn.Module) -> None:
        super().__init__()
        self.model = tbps_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


def _image_hw(config) -> tuple[int, int]:
    image_size = config.backbone.vision_config.image_size
    try:
        return int(image_size[0]), int(image_size[1])
    except (TypeError, IndexError):
        return int(image_size), int(image_size)


def _save_external(model: onnx.ModelProto, onnx_path: Path) -> None:
    """Re-save as .onnx + single .onnx.data (AI Hub directory upload format)."""
    data_name = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _print_op_counts(onnx_path: Path) -> dict[str, int]:
    m = onnx.load(str(onnx_path), load_external_data=False)
    opsets = [(op.domain or "ai.onnx", op.version) for op in m.opset_import]
    counts = Counter(n.op_type for n in m.graph.node)
    print(f"\nopset imports: {opsets}")
    print(f"total nodes: {sum(counts.values())}")
    highlight = [
        "Gelu", "Pow", "Tanh", "Erf", "LayerNormalization", "RMSNormalization",
        "ReduceMean", "Sqrt", "Reciprocal", "Mul", "Div", "Add", "Sub",
        "MatMul", "Softmax",
    ]
    print("op highlights:")
    for k in highlight:
        if k in counts:
            print(f"  {k:18s}: {counts[k]}")
    return dict(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export rotated vision encoder to ONNX opset-20.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_rotated",
    )
    p.add_argument(
        "--output-subdir",
        default="vision_onnx",
        help="Subdirectory under --model-dir to write the .onnx + .onnx.data.",
    )
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()

    print(f"Loading rotated model via load_rotated_model: {model_dir}")
    model, config = load_rotated_model(model_dir, args.device)

    h, w = _image_hw(config)
    dummy = torch.randn(1, 3, h, w)
    out_dir = model_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "vision_encoder.onnx"

    print(f"Exporting encode_image to ONNX opset {args.opset} (image {h}x{w}) ...")
    torch.onnx.export(
        VisionWrapper(model),
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["image_embedding"],
        dynamic_axes={"image": {0: "batch_size"}, "image_embedding": {0: "batch_size"}},
        opset_version=args.opset,
        dynamo=False,  # match the TorchScript export that produced exported_model_gelu20
    )

    # Re-save in the .onnx + single .onnx.data layout AI Hub expects.
    loaded = onnx.load(str(onnx_path), load_external_data=True)
    _save_external(loaded, onnx_path)
    print(f"Saved: {onnx_path}")

    _print_op_counts(onnx_path)
    print(
        "\nNext (GATE M2): \n"
        "  venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \\\n"
        f"    --onnx-model {onnx_path} \\\n"
        f"    --model-dir {model_dir} --rotated \\\n"
        "    --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32"
    )


if __name__ == "__main__":
    main()

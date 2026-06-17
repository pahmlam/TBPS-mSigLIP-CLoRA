#!/usr/bin/env python3
"""T0 — Export the text encoder to ONNX opset-20 (fused GELU).

Text counterpart of `export_rotated_vision_onnx.py`. The text encoder takes two
integer inputs (`input_ids`, `attention_mask`, shape [1, seq_len]) and returns the
768-d sentence embedding via `TBPS.encode_text({...})`. Opset 20 fuses the tanh-GELU
into a single `Gelu` op (avoiding the `Pow(x,3)` cubic that wrecks per-tensor
quantization), exactly as on the vision path.

Works for both the plain merged model and a (future) rotated text model — the
mean-preserving rotation keeps fused LayerNorm, so `_load_pytorch_model` reloads
either correctly.

GATE (T0): static-control text ONNX vs PyTorch cosine ~1.0 on vn3k_text_10
(run `compare_text_onnx_with_pytorch.py`). Op counts should show fused
`Gelu` and `LayerNormalization`, `Pow = 0`.

Example:
    venv/bin/python deployment/scripts/qnn/export_text_onnx.py \
      --model-dir artifacts/deployment/exports/exported_model
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

from compare_qnn_with_pytorch import _load_pytorch_model  # noqa: E402


class TextWrapper(nn.Module):
    """Wraps TBPS.encode_text for ONNX export (two integer inputs)."""

    def __init__(self, tbps_model: nn.Module) -> None:
        super().__init__()
        self.model = tbps_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )


def _seq_len(config) -> int:
    try:
        return int(config.tokenizer.model_max_length)
    except Exception:
        return 64


def _save_external(model: onnx.ModelProto, onnx_path: Path) -> None:
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
        "ReduceMean", "Sqrt", "Gather", "MatMul", "Softmax",
    ]
    print("op highlights:")
    for k in highlight:
        if k in counts:
            print(f"  {k:18s}: {counts[k]}")
    return dict(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export text encoder to ONNX opset-20.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model",
    )
    p.add_argument("--output-subdir", default="text_onnx")
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=0, help="0 = use config tokenizer max length.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()

    print(f"Loading model from {model_dir} ...")
    model, config = _load_pytorch_model(model_dir, "fp32", args.device)
    model.eval()

    seq_len = args.seq_len or _seq_len(config)
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

    out_dir = model_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "text_encoder.onnx"

    print(f"Exporting encode_text to ONNX opset {args.opset} (seq_len {seq_len}) ...")
    torch.onnx.export(
        TextWrapper(model),
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embedding": {0: "batch_size"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    loaded = onnx.load(str(onnx_path), load_external_data=True)
    _save_external(loaded, onnx_path)
    print(f"Saved: {onnx_path}")

    _print_op_counts(onnx_path)
    print(
        "\nNext (GATE T0): \n"
        "  venv/bin/python deployment/scripts/qnn/compare_text_onnx_with_pytorch.py \\\n"
        f"    --onnx-model {onnx_path} \\\n"
        f"    --model-dir {model_dir} \\\n"
        "    --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 \\\n"
        "    --json /tmp/text_static.json --csv /tmp/text_static.csv"
    )


if __name__ == "__main__":
    main()

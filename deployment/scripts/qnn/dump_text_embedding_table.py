#!/usr/bin/env python3
"""Dump the (rotated) text token-embedding table for on-device split-text encoding.

The split text encoder runs its transformer on HTP but needs the embedding lookup
`inputs_embeds = token_embedding[input_ids]` on a CPU (HTP v68 breaks the dynamic 250k
gather — see `w8a8_qat_rotated.md` §12A.8). This script extracts the rotated v8
`token_embedding.weight` and writes it as a standalone table that `board_text_encode.py`
reads on the RB3 CPU. It is a frozen model weight (Q already folded in); regenerate only
if the text model changes.

Two formats are written:
- **int8 per-row** (default deploy): `token_embedding_int8.bin` (~192 MB) + a per-row
  float32 scale `token_embedding_int8_scale.bin` (~1 MB) + `meta_int8.json`. Row `r` is
  reconstructed as `int8[r] * scale[r]`. Quality is effectively unchanged because the
  split bin re-quantizes `inputs_embeds` to uint8 at its I/O anyway (the table's per-row
  int8 is finer than the bin's per-tensor input quant).
- **fp16** (simple/alternative): `token_embedding_fp16.bin` (~384 MB) + `meta.json`.

Run on the host once at deploy-build time; push the chosen table to the board.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_qnn_with_pytorch import _load_pytorch_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump rotated text token-embedding table (int8 + fp16).")
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/bin/token_embedding_v8")
    p.add_argument("--formats", nargs="+", choices=["int8", "fp16"], default=["int8", "fp16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_dir} ...")
    model, _ = _load_pytorch_model(model_dir, "fp32", "cpu")
    model.eval()
    W = model.text_model.embeddings.token_embedding.weight.detach().float().cpu().numpy()  # [V, D] rotated
    vocab, dim = W.shape
    print(f"token_embedding: vocab={vocab} dim={dim} (rotated, Q folded)")

    if "int8" in args.formats:
        scale = (np.abs(W).max(axis=1, keepdims=True) / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0
        w_int8 = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
        w_int8.tofile(out / "token_embedding_int8.bin")
        scale.tofile(out / "token_embedding_int8_scale.bin")
        (out / "meta_int8.json").write_text(json.dumps({
            "vocab": int(vocab), "dim": int(dim), "dtype": "int8_perrow",
            "weight_file": "token_embedding_int8.bin", "scale_file": "token_embedding_int8_scale.bin",
            "source": str(model_dir), "rotated": True,
            "note": "per-row symmetric int8: row r = int8[r]*scale[r]",
        }, indent=2))
        print(f"  int8 per-row: {w_int8.nbytes / 1e6:.0f} MB + scale {scale.nbytes / 1e6:.1f} MB")

    if "fp16" in args.formats:
        W.astype(np.float16).tofile(out / "token_embedding_fp16.bin")
        (out / "meta.json").write_text(json.dumps({
            "vocab": int(vocab), "dim": int(dim), "dtype": "float16",
            "source": str(model_dir), "rotated": True,
            "note": "lookup: inputs_embeds = table[input_ids]",
        }, indent=2))
        print(f"  fp16: {W.astype(np.float16).nbytes / 1e6:.0f} MB")

    print(f"Wrote table(s) -> {out}")


if __name__ == "__main__":
    main()

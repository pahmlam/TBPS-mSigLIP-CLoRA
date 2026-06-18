#!/usr/bin/env python3
"""T1 - Mean-preserving rotation equalization for the text encoder.

Text counterpart of `rotate_vision_encoder.py`. T0 profiling
(`profile_text_activation_outliers.py`) showed the text residual stream is as
spiky as vision (concentration 200-404x on `layers.N/Add`, `fc2/Add`), so plain
per-tensor W8A8 will collapse the same way. We apply the same fix: rotate the
residual stream by an orthogonal, mean-preserving Q (Q.1 = 1) folded offline into
the weights, so the FP32 sentence embedding is unchanged but no single channel
dominates the per-tensor scale.

WHY mean-preserving (Q.1 = 1) and not a free rotation
-----------------------------------------------------
The three norms (layer_norm1/2, final_layer_norm) stay as fused LayerNorm ops
(set to identity affine after folding their affine into the readers). An
identity-affine LayerNorm commutes with Q **iff** Q fixes the all-ones axis:

    Q.1 = 1, Q orthogonal  =>  mean(Qx) = mean(x), std(Qx) = std(x)
    =>  LN_id(Qx) = Q . LN_id(x)        (rotation passes through the norm)

The reader's folded Q^T then cancels it: output-invariant. Keeping the fused
LayerNorm (vs decomposing to RMSNorm) avoids exposing Pow/ReduceMean/Div to
per-tensor quantization - the M3 failure mode seen on vision.

Residual map (SiglipTextTransformer)
------------------------------------
Writers (residual producers), fold  W <- Q W,  b <- Q b  (embeddings: P <- P Q^T):
  - embeddings.token_embedding      (nn.Embedding row writer)
  - embeddings.position_embedding   (nn.Embedding row writer)
  - every layer self_attn.out_proj
  - every layer mlp.fc2
Readers (residual consumers), fold  W <- W Q^T:
  - every layer q/k/v_proj          (read layer_norm1)
  - every layer mlp.fc1             (read layer_norm2)
  - head                            (reads final_layer_norm last-token pooler)

No patch-conv and no MHA pooling head (text head is a plain Linear), so this is
strictly simpler than the vision rotation. `backbone.text_projection` (if any)
sits in the un-rotated head-output space and is untouched.

GATE (T1, free/local): encode_text FP32 cosine vs the original model must be
~1.0 on vn3k_text_10. Below threshold => a fold bug (affine fold order, Q vs Q^T,
or embedding row-fold transpose).

Example:
    venv/bin/python deployment/scripts/qnn/rotate_text_encoder.py \\
      --model-dir artifacts/deployment/exports/exported_model \\
      --output-dir artifacts/deployment/exports/exported_model_text_rotated \\
      --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the exact loaders + low-level folds the vision rotation uses so the math
# is identical and the gate is measured on the same model loader as deployment.
from compare_qnn_with_pytorch import _load_pytorch_model  # noqa: E402
from compare_text_onnx_with_pytorch import (  # noqa: E402
    _parse_dual_input_list,
    _read_raw_ints,
)
from rotate_vision_encoder import (  # noqa: E402
    _cosine_stats,
    _ln_affine,
    _mean_preserving_orthogonal,
    _set_ln_identity,
    fold_affine_into_reader,
    fold_left_into_linear_writer,
    fold_right_into_embedding_writer,
    fold_right_into_reader,
)


# --------------------------------------------------------------------------- #
# Phase A: fold each LN affine into its readers, set LN identity (keep fused).
# --------------------------------------------------------------------------- #
def phase_a(text_model: nn.Module, dim: int) -> dict:
    layers = text_model.encoder.layers
    n_layers = len(layers)

    for layer in layers:
        g1, b1 = _ln_affine(layer.layer_norm1)
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            fold_affine_into_reader(proj, g1, b1)
        _set_ln_identity(layer.layer_norm1)

        g2, b2 = _ln_affine(layer.layer_norm2)
        fold_affine_into_reader(layer.mlp.fc1, g2, b2)
        _set_ln_identity(layer.layer_norm2)

    gf, bf = _ln_affine(text_model.final_layer_norm)
    fold_affine_into_reader(text_model.head, gf, bf)
    _set_ln_identity(text_model.final_layer_norm)

    return {"n_layers": n_layers, "ln_made_identity": 2 * n_layers + 1, "dim": dim, "kept_fused_layernorm": True}


# --------------------------------------------------------------------------- #
# Phase B: rotate residual stream by mean-preserving orthogonal Q.
# --------------------------------------------------------------------------- #
def phase_b(text_model: nn.Module, Q: torch.Tensor) -> dict:
    Qt = Q.t().contiguous()
    layers = text_model.encoder.layers

    # Writers: residual producers. Embeddings add looked-up rows -> P <- P Q^T.
    fold_right_into_embedding_writer(text_model.embeddings.token_embedding, Qt)
    fold_right_into_embedding_writer(text_model.embeddings.position_embedding, Qt)
    for layer in layers:
        fold_left_into_linear_writer(layer.self_attn.out_proj, Q)
        fold_left_into_linear_writer(layer.mlp.fc2, Q)

    # Readers: residual consumers. W <- W Q^T.
    for layer in layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            fold_right_into_reader(proj, Qt)
        fold_right_into_reader(layer.mlp.fc1, Qt)
    fold_right_into_reader(text_model.head, Qt)

    err = (Q @ Qt - torch.eye(Q.shape[0], dtype=torch.float64, device=Q.device)).abs().max().item()
    return {"orthogonality_max_err": err}


# --------------------------------------------------------------------------- #
# Gate: encode_text cosine vs reference (two integer inputs per sample).
# --------------------------------------------------------------------------- #
def _seq_len(config, fallback: int = 64) -> int:
    try:
        return int(config.tokenizer.model_max_length)
    except Exception:
        return fallback


def _embed_text(model: nn.Module, rows: list[dict[str, Path]], seq_len: int, device: str) -> torch.Tensor:
    out_rows = []
    with torch.no_grad():
        for entry in rows:
            ids = _read_raw_ints(entry["input_ids"], seq_len)
            attn = _read_raw_ints(entry["attention_mask"], seq_len)
            caption_input = {
                "input_ids": torch.from_numpy(ids.astype(np.int64)).to(device),
                "attention_mask": torch.from_numpy(attn.astype(np.int64)).to(device),
            }
            emb = model.encode_text(caption_input).detach().float().cpu().reshape(-1)
            out_rows.append(emb)
    return torch.stack(out_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mean-preserving rotation of the text encoder (T1).")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated",
    )
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10")
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--precision", choices=["fp32"], default="fp32", help="Rotation math requires fp32.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=0, help="0 = use config tokenizer max length.")
    p.add_argument("--seed", type=int, default=2400, help="Seed for the orthogonal Q.")
    p.add_argument("--gate-threshold", type=float, default=0.9999)
    p.add_argument("--skip-phase-b", action="store_true", help="Apply only Phase A (debug the affine fold).")
    p.add_argument("--no-save", action="store_true", help="Run the gate but do not write the rotated model.")
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated/rotation_summary.json",
    )
    return p.parse_args()


def _save_rotated(model: nn.Module, src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "config.yaml", out_dir / "config.yaml")
    torch.save(model.state_dict(), out_dir / "model_fp32.pt")


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()

    rows = _parse_dual_input_list(input_dir, args.input_list)
    print(f"Loading PyTorch model from {model_dir} ...")
    model, config = _load_pytorch_model(model_dir, args.precision, args.device)
    model.eval()

    text_model = model.text_model
    dim = text_model.final_layer_norm.normalized_shape[0]
    seq_len = args.seq_len or _seq_len(config)
    print(f"Text residual dim = {dim}; encoder layers = {len(text_model.encoder.layers)}; seq_len = {seq_len}")

    print(f"Computing reference embeddings on {len(rows)} inputs ...")
    ref_emb = _embed_text(model, rows, seq_len, args.device)

    print("Phase A: fold LN affine into readers + set LN identity (keep fused LayerNorm) ...")
    a_info = phase_a(text_model, dim)
    after_a = _embed_text(model, rows, seq_len, args.device)
    cos_a = _cosine_stats(ref_emb, after_a)
    print(f"  Phase A invariance cosine: mean={cos_a['mean']:.8f} min={cos_a['min']:.8f}")

    b_info: dict = {"applied": False}
    cos_b = cos_a
    if not args.skip_phase_b:
        print(f"Phase B: rotate residual by mean-preserving Q (seed={args.seed}) ...")
        Q = _mean_preserving_orthogonal(dim, args.seed)
        ones = torch.ones(dim, dtype=torch.float64)
        mean_err = (Q @ ones - ones).abs().max().item()
        b_info = {"applied": True, "mean_preserve_max_err": mean_err, **phase_b(text_model, Q)}
        print(f"  Q orthogonality max err: {b_info['orthogonality_max_err']:.2e}; Q@1=1 max err: {mean_err:.2e}")
        after_b = _embed_text(model, rows, seq_len, args.device)
        cos_b = _cosine_stats(ref_emb, after_b)
        print(f"  Phase B invariance cosine: mean={cos_b['mean']:.8f} min={cos_b['min']:.8f}")

    passed = cos_b["min"] >= args.gate_threshold
    summary = {
        "model_dir": str(model_dir),
        "input_dir": str(input_dir),
        "num_samples": len(rows),
        "dim": dim,
        "seq_len": seq_len,
        "seed": args.seed,
        "gate_threshold": args.gate_threshold,
        "phase_a": {**a_info, "cosine": cos_a},
        "phase_b": {**b_info, "cosine": cos_b},
        "gate_passed": passed,
    }

    if passed and not args.no_save:
        out_dir = args.output_dir.expanduser().resolve()
        print(f"GATE PASS -> saving rotated text model to {out_dir}")
        _save_rotated(model, model_dir, out_dir)
        summary["output_dir"] = str(out_dir)

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== TEXT ROTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if not passed:
        print(
            "\nGATE FAIL: encode_text cosine below threshold. Likely fold bug "
            "(affine fold order, Q vs Q^T, or embedding row-fold transpose). Model NOT saved."
        )
        raise SystemExit(1)
    print("\nGATE PASS: FP32 output-invariant. Ready to export rotated text ONNX (opset-20) + QAT.")


if __name__ == "__main__":
    main()

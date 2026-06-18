#!/usr/bin/env python3
"""Learned mean-preserving rotation (SpinQuant-inspired) for the TEXT encoder.

Text counterpart of `learn_rotation.py`. Replaces the *random* mean-preserving
orthogonal Q of `rotate_text_encoder.py` (T1) with a **learned** one that directly
minimizes the per-tensor INT8 difficulty of the rotated text residual activations.

Same objective and parametrization as the vision version (see `learn_rotation.py`
for the full derivation): minimize `Σ_tensor (max-abs of a·Q^T)^2` over the
rotation-site activations, with Q mean-preserving (Q·1 = 1, keeps fused LayerNorm)
via the Cayley parametrization `Q = U · blockdiag(1, Cayley(skew)) · U^T`.

The ONLY differences from the vision version are the data path (int input_ids +
attention_mask instead of NCHW images), the rotation sites (text encoder
`layer_norm1/2`, `out_proj`, `fc2`, plus `final_layer_norm` instead of
`post_layernorm`), and the gate metric (`encode_text` invariance). The learned Q
is folded by the exact text folds of `rotate_text_encoder.py`, so the output is a
drop-in replacement for `exported_model_text_rotated` and feeds the same
export → QAT (`--modality text`) → W8A8 → eval pipeline.

GATE: after folding the learned Q, encode_text FP32 cosine vs the original model
must be ~1.0 (phase_a + phase_b are output-invariant by construction).

Example:
    venv/bin/python deployment/scripts/qnn/learn_rotation_text.py \
      --model-dir artifacts/deployment/exports/exported_model \
      --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned \
      --calib-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500 \
      --gate-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 \
      --num-calib 256 --tokens-per-sample 32 --steps 3000
"""

from __future__ import annotations

import argparse
import json
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

from compare_qnn_with_pytorch import _load_pytorch_model  # noqa: E402
from compare_text_onnx_with_pytorch import (  # noqa: E402
    _parse_dual_input_list,
    _read_raw_ints,
)
from learn_rotation import MeanPreservingRotation, _per_tensor_max_sq  # noqa: E402
from rotate_text_encoder import (  # noqa: E402
    _embed_text,
    _save_rotated,
    _seq_len,
    phase_a,
    phase_b,
)
from rotate_vision_encoder import _cosine_stats  # noqa: E402


# --------------------------------------------------------------------------- #
# Cache the residual-stream activations the (folded) rotation will quantize.
# Text rotation sites: each layer_norm1/2 output (reader inputs), each
# self_attn.out_proj / mlp.fc2 output (writer outputs), and final_layer_norm.
# --------------------------------------------------------------------------- #
def _collect_rotation_site_activations(
    model: nn.Module,
    text_model: nn.Module,
    rows: list[dict[str, Path]],
    seq_len: int,
    device: str,
    tokens_per_sample: int,
    seed: int,
) -> list[torch.Tensor]:
    """Return a list of (rows, dim) tensors, one per rotation-affected site.

    Padded positions are kept on purpose: per-tensor INT8 scale on-device covers
    the whole activation tensor (including pad rows), so the learned Q must shrink
    their max-abs too. Tokens are subsampled per caption to bound memory.
    """
    layers = text_model.encoder.layers
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []

    def _hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captured.setdefault(name, []).append(t.detach().reshape(-1, t.shape[-1]).float().cpu())
        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.layer_norm1.register_forward_hook(_hook(f"l{i}.ln1")))
        handles.append(layer.layer_norm2.register_forward_hook(_hook(f"l{i}.ln2")))
        handles.append(layer.self_attn.out_proj.register_forward_hook(_hook(f"l{i}.out_proj")))
        handles.append(layer.mlp.fc2.register_forward_hook(_hook(f"l{i}.fc2")))
    handles.append(text_model.final_layer_norm.register_forward_hook(_hook("final_ln")))

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for entry in rows:
            ids = _read_raw_ints(entry["input_ids"], seq_len)
            attn = _read_raw_ints(entry["attention_mask"], seq_len)
            caption_input = {
                "input_ids": torch.from_numpy(ids.astype(np.int64)).to(device),
                "attention_mask": torch.from_numpy(attn.astype(np.int64)).to(device),
            }
            model.encode_text(caption_input)
            for name, lst in captured.items():
                t = lst[-1]
                if tokens_per_sample and t.shape[0] > tokens_per_sample:
                    idx = torch.randperm(t.shape[0], generator=gen)[:tokens_per_sample]
                    lst[-1] = t[idx].contiguous()

    for h in handles:
        h.remove()

    return [torch.cat(lst, dim=0) for lst in captured.values()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn a mean-preserving rotation for the text encoder (SpinQuant-style).")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned",
    )
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_calib_500",
    )
    p.add_argument(
        "--gate-input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10",
    )
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--num-calib", type=int, default=256, help="Calibration captions for caching activations.")
    p.add_argument("--tokens-per-sample", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=0, help="0 = use config tokenizer max length.")
    p.add_argument("--seed", type=int, default=2400)
    p.add_argument("--gate-threshold", type=float, default=0.9999)
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned/learned_rotation_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    device = args.device

    print(f"Loading model from {model_dir} ...")
    model, config = _load_pytorch_model(model_dir, "fp32", device)
    model.eval()
    text_model = model.text_model
    dim = text_model.final_layer_norm.normalized_shape[0]
    seq_len = args.seq_len or _seq_len(config)
    print(f"Text residual dim = {dim}; layers = {len(text_model.encoder.layers)}; seq_len = {seq_len}")

    gate_rows = _parse_dual_input_list(args.gate_input_dir.expanduser().resolve(), args.input_list)
    print("Computing reference embeddings (pre-rotation) ...")
    ref_emb = _embed_text(model, gate_rows, seq_len, device)

    print("Phase A: fold LN affine into readers, set LN identity ...")
    phase_a(text_model, dim)

    calib_rows = _parse_dual_input_list(args.calib_dir.expanduser().resolve(), args.input_list)
    if args.num_calib > 0:
        calib_rows = calib_rows[: args.num_calib]
    print(f"Caching rotation-site activations on {len(calib_rows)} calib captions ...")
    acts = _collect_rotation_site_activations(
        model, text_model, calib_rows, seq_len, device, args.tokens_per_sample, args.seed
    )
    # Optimize in float32 (GPU float64 is throttled ~1/32-1/64). The fold + gate
    # below recompute Q in float64 for exact orthogonality.
    acts = [a.to(torch.float32).to(device) for a in acts]
    print(f"  cached {len(acts)} site tensors; rows each ~{acts[0].shape[0]}")

    with torch.no_grad():
        base_obj = sum(_per_tensor_max_sq(a) for a in acts).item()
        base_maxabs = max(a.abs().max().item() for a in acts)

    rot = MeanPreservingRotation(dim, args.seed, device)
    opt = torch.optim.Adam(rot.parameters(), lr=args.lr)

    print(f"Optimizing learned rotation: {args.steps} steps (base obj={base_obj:.1f}, max|a|={base_maxabs:.1f}) ...")
    for step in range(args.steps):
        opt.zero_grad()
        Qt = rot.matrix(torch.float32).t()  # float32 optimization path
        obj = 0.0
        for a in acts:
            obj = obj + _per_tensor_max_sq(a @ Qt)  # rotated reader/writer activation = a · Q^T
        obj.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                cur_max = max((a @ rot.matrix(torch.float32).t()).abs().max().item() for a in acts)
            print(f"  step {step:4d}  obj={float(obj):.1f}  max|Qa|={cur_max:.2f}")

    with torch.no_grad():
        Q = rot.matrix(torch.float64).detach()  # precise Q for fold + orthogonality
        orth_err = (Q @ Q.t() - torch.eye(dim, dtype=Q.dtype, device=device)).abs().max().item()
        ones = torch.ones(dim, dtype=Q.dtype, device=device)
        mean_err = (Q @ ones - ones).abs().max().item()
        Q32t = rot.matrix(torch.float32).t()
        final_obj = sum(_per_tensor_max_sq(a @ Q32t) for a in acts).item()
        final_maxabs = max((a @ Q32t).abs().max().item() for a in acts)

    print(f"Learned Q: orth_err={orth_err:.2e} mean_err={mean_err:.2e}")
    print(f"Objective {base_obj:.1f} -> {final_obj:.1f}  | max|a| {base_maxabs:.2f} -> {final_maxabs:.2f}")

    print("Phase B: fold learned Q into weights ...")
    phase_b(text_model, Q.to(torch.float64))

    after = _embed_text(model, gate_rows, seq_len, device)
    cos = _cosine_stats(ref_emb, after)
    print(f"Output-invariance cosine: mean={cos['mean']:.8f} min={cos['min']:.8f}")
    passed = cos["min"] >= args.gate_threshold

    summary = {
        "model_dir": str(model_dir),
        "dim": dim,
        "seq_len": seq_len,
        "num_calib": len(calib_rows),
        "tokens_per_sample": args.tokens_per_sample,
        "steps": args.steps,
        "lr": args.lr,
        "base_objective": base_obj,
        "final_objective": final_obj,
        "base_max_abs": base_maxabs,
        "final_max_abs": final_maxabs,
        "orthogonality_max_err": orth_err,
        "mean_preserve_max_err": mean_err,
        "invariance_cosine": cos,
        "gate_passed": passed,
    }

    if passed and not args.no_save:
        out_dir = args.output_dir.expanduser().resolve()
        print(f"GATE PASS -> saving learned-rotated text model to {out_dir}")
        _save_rotated(model, model_dir, out_dir)
        summary["output_dir"] = str(out_dir)

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== LEARNED TEXT ROTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("GATE FAIL: output not invariant after folding learned Q (fold/parametrization bug).")
    print("\nGATE PASS. Next: export_text_onnx.py on the output dir, then QAT (--modality text)/quantize.")


if __name__ == "__main__":
    main()

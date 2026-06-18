#!/usr/bin/env python3
"""M1 - Rotation-based equalization (QuaRot/SliceGPT-style) for the vision encoder.

Plan: `deployment/docs/journal/[deploy]-2026-06-15.md` (section 5) and
`[deploy-plan]-2026-06-14.md`.

WHY
---
On HTP v68, 16-bit activations (A16) are rejected for LayerNorm and attention
act x act matmul (need v73+). Per-tensor W8A8 is the only deployable INT8 path,
but it collapses because a few fixed residual-stream channels carry huge values
(concentration ~876x). Clipping fails - those outliers carry real signal.

Rotation removes the *per-tensor* pressure without clipping: rotate the residual
stream by an orthogonal Q. Q smears the energy of the spiky channels across all
768 dims, so no single channel dominates the per-tensor scale, yet the rotation
is folded into weights offline so the FP32 output is unchanged.

For an orthogonal Q to commute through the normalization layers, the norms must
be RMSNorm (no mean-subtraction), because:

    RMSNorm(Qx) = Qx / rms(x) = Q * RMSNorm(x)          (commutes)
    LayerNorm(Qx) != Q * LayerNorm(x)                   (mean(Qx) != 0 in general)

So the transform is two phases, both output-invariant in FP32:

Phase A - LayerNorm -> RMSNorm
  1. Fold each LN affine (gamma, beta) into its consumer linears (readers), set
     the LN affine to identity.
  2. Fold M = I - (1/d) 11^T into every writer of the residual stream
     (embeddings, every out_proj, every fc2) so the residual is mean-zero. Since
     LayerNorm subtracts mean anyway, LN(Mx) = LN(x): output-invariant.
  3. Replace the (now identity-affine, mean-zero-input) LayerNorm modules with
     RMSNorm. Same value, but the op no longer subtracts mean.

Phase B - rotate residual by orthogonal Q (768x768)
  - Writers (embeddings, out_proj, fc2):   W <- Q W,   b <- Q b
  - Readers (q/k/v <- LN1, fc1 <- LN2, head.attn K/V <- post_LN): W <- W Q^T
  RMSNorm(Qx) = Q RMSNorm(x) carries the rotation through the norm, and the
  reader's Q^T cancels it: output-invariant.

The head pooling block lives in the *un-rotated* space; the rotation is undone at
the head.attention K/V boundary (fold Q^T there). The probe is learned (query),
out_proj/layernorm/mlp of the head are untouched.

GATE (M1, free/local): encode_image FP32 cosine vs the original model must be
~1.0 on vn3k_test_10. Anything below means a fold bug (usually mean-subtraction
fold order, affine fold order, or Q vs Q^T).

Example:
    venv/bin/python deployment/scripts/qnn/rotate_vision_encoder.py \\
      --model-dir artifacts/deployment/exports/exported_model \\
      --output-dir artifacts/deployment/exports/exported_model_rotated \\
      --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the exact model + raw-input loaders the compare/benchmark scripts use so
# the gate is measured on the same tensors as the deployment pipeline.
from compare_qnn_with_pytorch import (  # noqa: E402
    _load_pytorch_model,
    _parse_input_list,
    _read_raw_tensor,
)


# --------------------------------------------------------------------------- #
# RMSNorm (no mean-subtraction). Value-equal to a mean-zero, identity-affine
# LayerNorm, but commutes with an orthogonal rotation.
# --------------------------------------------------------------------------- #
class RMSNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim), assumed (numerically) mean-zero along the last dim.
        ms = x.pow(2).mean(dim=-1, keepdim=True)  # (..., 1)
        return x * torch.rsqrt(ms + self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# --------------------------------------------------------------------------- #
# Low-level folds (all math in float64, then cast back to the param dtype).
# --------------------------------------------------------------------------- #
def _as64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


def _set_param(p: nn.Parameter, value64: torch.Tensor) -> None:
    p.data.copy_(value64.to(p.dtype))


def fold_affine_into_reader(linear: nn.Linear, gamma: torch.Tensor, beta: torch.Tensor) -> None:
    """Reader consumes LN(x) = gamma*x_hat + beta. Absorb affine into the linear.

    y = LN(x) W^T + b = x_hat (W diag(gamma))^T + (b + W beta)
    => W <- W diag(gamma)   (scale column j by gamma_j)
       b <- b + W_orig beta
    """
    W = _as64(linear.weight)  # (out, in)
    g = _as64(gamma)  # (in,)
    bias_add = W @ _as64(beta)  # (out,)
    _set_param(linear.weight, W * g.unsqueeze(0))
    if linear.bias is not None:
        _set_param(linear.bias, _as64(linear.bias) + bias_add)
    else:
        linear.bias = nn.Parameter(bias_add.to(linear.weight.dtype))


def fold_left_into_linear_writer(linear: nn.Linear, T: torch.Tensor) -> None:
    """Writer output (column vec) is mapped y -> T y. W <- T W, b <- T b."""
    W = _as64(linear.weight)  # (out, in)
    _set_param(linear.weight, T @ W)
    if linear.bias is not None:
        _set_param(linear.bias, T @ _as64(linear.bias))


def fold_right_into_reader(linear: nn.Linear, T: torch.Tensor) -> None:
    """Reader input (rotated) -> undo with W <- W T (T = Q^T). Bias unchanged."""
    W = _as64(linear.weight)  # (out, in)
    _set_param(linear.weight, W @ T)


def fold_left_into_conv_writer(conv: nn.Conv2d, T: torch.Tensor) -> None:
    """Conv output channels (768) are the residual dim. Mix them by T."""
    W = _as64(conv.weight)  # (out, in, kh, kw)
    out_ch = W.shape[0]
    flat = W.reshape(out_ch, -1)  # (out, in*kh*kw)
    _set_param(conv.weight, (T @ flat).reshape_as(W))
    if conv.bias is not None:
        _set_param(conv.bias, T @ _as64(conv.bias))


def fold_right_into_embedding_writer(emb: nn.Embedding, T: torch.Tensor) -> None:
    """Position embedding rows are added to the residual (row vec r -> r T^T).

    Stored weight P has shape (num_pos, dim); rotating each row r -> T r means
    P <- P T^T. Callers pass T already transposed (M is symmetric; for Q pass
    Q^T so the row op becomes r -> r Q^T == (Q r)^T, i.e. r -> Q r).
    """
    P = _as64(emb.weight)  # (num_pos, dim)
    _set_param(emb.weight, P @ T)


# --------------------------------------------------------------------------- #
# nn.MultiheadAttention helpers (packed in_proj_weight = [Wq; Wk; Wv]).
# K and V read the post_layernorm output (a residual reader); Q reads the probe.
# --------------------------------------------------------------------------- #
def _mha_kv_slices(mha: nn.MultiheadAttention) -> tuple[slice, slice]:
    d = mha.embed_dim
    return slice(d, 2 * d), slice(2 * d, 3 * d)


def fold_affine_into_mha_kv(mha: nn.MultiheadAttention, gamma: torch.Tensor, beta: torch.Tensor) -> None:
    """Absorb post_LN affine into the K and V parts of the packed in_proj."""
    k_sl, v_sl = _mha_kv_slices(mha)
    W = _as64(mha.in_proj_weight)  # (3d, d)
    g = _as64(gamma)
    b64 = _as64(beta)
    new_W = W.clone()
    new_b = _as64(mha.in_proj_bias).clone() if mha.in_proj_bias is not None else None
    for sl in (k_sl, v_sl):
        Wpart = W[sl]  # (d, d)
        if new_b is not None:
            new_b[sl] = new_b[sl] + Wpart @ b64
        new_W[sl] = Wpart * g.unsqueeze(0)
    _set_param(mha.in_proj_weight, new_W)
    if new_b is not None:
        _set_param(mha.in_proj_bias, new_b)


def fold_right_into_mha_kv(mha: nn.MultiheadAttention, T: torch.Tensor) -> None:
    """Undo rotation on K/V reader: W_kv <- W_kv T (T = Q^T). Bias unchanged."""
    k_sl, v_sl = _mha_kv_slices(mha)
    W = _as64(mha.in_proj_weight)
    new_W = W.clone()
    for sl in (k_sl, v_sl):
        new_W[sl] = W[sl] @ T
    _set_param(mha.in_proj_weight, new_W)


# --------------------------------------------------------------------------- #
# LayerNorm -> RMSNorm replacement
# --------------------------------------------------------------------------- #
def _ln_affine(ln: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    d = ln.normalized_shape[0]
    gamma = ln.weight.detach().clone() if ln.weight is not None else torch.ones(d)
    beta = ln.bias.detach().clone() if ln.bias is not None else torch.zeros(d)
    return gamma, beta


def _set_ln_identity(ln: nn.LayerNorm) -> None:
    """Make a LayerNorm identity-affine (gamma=1, beta=0) but KEEP it as a fused
    LayerNorm op. The affine has already been folded into the readers.

    Keeping the fused LayerNormalization op (vs decomposing to RMSNorm) is what
    avoids exposing Pow(x^2)/ReduceMean/Div internals to per-tensor quantization
    - the M3 failure mode. Requires a mean-preserving rotation Q (Q.1 = 1) so
    LayerNorm (which subtracts mean) still commutes with the rotation.
    """
    with torch.no_grad():
        if ln.weight is not None:
            ln.weight.fill_(1.0)
        if ln.bias is not None:
            ln.bias.zero_()


def _replace_ln_with_rmsnorm(parent: nn.Module, attr: str, ln: nn.LayerNorm) -> None:
    dim = ln.normalized_shape[0]
    setattr(parent, attr, RMSNormNoAffine(dim, float(ln.eps)))


def _swap_vision_norms_to_rmsnorm(vision_model: nn.Module) -> None:
    """Replace the rotated norms (layer_norm1/2 + post_layernorm) with RMSNorm.

    Only swaps the module type; the folded weights live in the linears and are
    loaded separately. head.layernorm is intentionally left as LayerNorm (it sits
    in the un-rotated space and keeps its params).
    """
    for layer in vision_model.encoder.layers:
        _replace_ln_with_rmsnorm(layer, "layer_norm1", layer.layer_norm1)
        _replace_ln_with_rmsnorm(layer, "layer_norm2", layer.layer_norm2)
    _replace_ln_with_rmsnorm(vision_model, "post_layernorm", vision_model.post_layernorm)


def load_rotated_model(model_dir: Path, device: str = "cpu"):
    """Load a rotated model saved by this script.

    The mean-preserving rotation keeps the fused LayerNorm op (set to identity
    affine), so the saved state_dict has standard LayerNorm params and the normal
    loader reloads it correctly - no module swap needed. Kept as a named entry
    point so downstream (M2 export, compares) has a stable import.
    """
    model, config = _load_pytorch_model(Path(model_dir).expanduser().resolve(), "fp32", device)
    model.eval()
    return model, config


# --------------------------------------------------------------------------- #
# Phase A: LayerNorm -> RMSNorm (affine fold + mean-subtraction fold + replace)
# --------------------------------------------------------------------------- #
def phase_a(vision_model: nn.Module, dim: int) -> dict:
    """Fold each LN affine into its readers and set the LN to identity affine.

    The fused LayerNorm op is KEPT (not replaced by RMSNorm). With a mean-
    preserving rotation Q (Q.1 = 1), an identity-affine LayerNorm commutes with
    the rotation, so we avoid the M3 failure where decomposed RMSNorm
    (Pow/ReduceMean/Div) exposed normalization internals to quantization.
    No mean-subtraction fold is needed: LayerNorm still subtracts the mean.
    """
    layers = vision_model.encoder.layers
    n_layers = len(layers)

    for layer in layers:
        g1, b1 = _ln_affine(layer.layer_norm1)
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            fold_affine_into_reader(proj, g1, b1)
        _set_ln_identity(layer.layer_norm1)

        g2, b2 = _ln_affine(layer.layer_norm2)
        fold_affine_into_reader(layer.mlp.fc1, g2, b2)
        _set_ln_identity(layer.layer_norm2)

    gp, bp = _ln_affine(vision_model.post_layernorm)
    fold_affine_into_mha_kv(vision_model.head.attention, gp, bp)
    _set_ln_identity(vision_model.post_layernorm)

    return {"n_layers": n_layers, "ln_made_identity": 2 * n_layers + 1, "dim": dim, "kept_fused_layernorm": True}


# --------------------------------------------------------------------------- #
# Phase B: rotate residual stream by orthogonal Q
# --------------------------------------------------------------------------- #
def _mean_preserving_orthogonal(dim: int, seed: int) -> torch.Tensor:
    """Orthogonal Q that fixes the all-ones direction: Q @ 1 = 1.

    Built as Q = U R U^T where U[:,0] = 1/sqrt(dim) and R = blockdiag(1, R_c) with
    R_c a random orthogonal on the (dim-1) complement. Q is identity on the mean
    axis and a random rotation on the orthogonal complement (where channel
    outliers live), so mean(Qx) = mean(x) and std(Qx) = std(x): an identity-affine
    LayerNorm commutes with Q, letting us keep the fused LayerNorm op.
    """
    g = torch.Generator().manual_seed(seed)
    v0 = torch.ones(dim, dtype=torch.float64) / (dim ** 0.5)

    # Orthonormal basis U with first column == v0.
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    A[:, 0] = v0
    U, _ = torch.linalg.qr(A)
    if torch.dot(U[:, 0], v0) < 0:
        U[:, 0] = -U[:, 0]

    # Random orthogonal on the complement; identity on the mean axis.
    B = torch.randn(dim - 1, dim - 1, generator=g, dtype=torch.float64)
    Rc, rr = torch.linalg.qr(B)
    Rc = Rc * torch.sign(torch.diagonal(rr)).unsqueeze(0)  # deterministic sign
    R = torch.eye(dim, dtype=torch.float64)
    R[1:, 1:] = Rc

    Q = U @ R @ U.t()
    return Q


def phase_b(vision_model: nn.Module, Q: torch.Tensor) -> dict:
    Qt = Q.t().contiguous()
    layers = vision_model.encoder.layers

    # Writers: W <- Q W, b <- Q b.
    fold_left_into_conv_writer(vision_model.embeddings.patch_embedding, Q)
    fold_right_into_embedding_writer(vision_model.embeddings.position_embedding, Qt)  # row r -> r Q^T = (Q r)^T
    for layer in layers:
        fold_left_into_linear_writer(layer.self_attn.out_proj, Q)
        fold_left_into_linear_writer(layer.mlp.fc2, Q)

    # Readers: W <- W Q^T.
    for layer in layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            fold_right_into_reader(proj, Qt)
        fold_right_into_reader(layer.mlp.fc1, Qt)
    fold_right_into_mha_kv(vision_model.head.attention, Qt)

    # Orthogonality diagnostic.
    err = (Q @ Qt - torch.eye(Q.shape[0], dtype=torch.float64, device=Q.device)).abs().max().item()
    return {"orthogonality_max_err": err}


# --------------------------------------------------------------------------- #
# Phase C (R2): rotate the per-head value / attention-output space (head_dim)
# --------------------------------------------------------------------------- #
def _hadamard(n: int) -> torch.Tensor:
    """Normalized Sylvester Hadamard, n a power of two: H H^T = I (float64)."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard requires a power-of-two size, got {n}")
    H = torch.ones((1, 1), dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H / (n ** 0.5)


def _blockdiag_headdim_rotation(num_heads: int, head_dim: int) -> tuple[torch.Tensor, bool]:
    """Block-diagonal orthogonal on the concatenated-heads axis (one H per head).

    Hadamard when head_dim is a power of two (the optimal incoherent spreader),
    else a deterministic random orthogonal fallback. No mean-preservation needed:
    this rotation lives on head_dim and never passes through a LayerNorm.
    """
    if head_dim & (head_dim - 1) == 0:
        H, is_hadamard = _hadamard(head_dim), True
    else:
        g = torch.Generator().manual_seed(0)
        A = torch.randn(head_dim, head_dim, generator=g, dtype=torch.float64)
        H, rr = torch.linalg.qr(A)
        H = H * torch.sign(torch.diagonal(rr)).unsqueeze(0)
        is_hadamard = False
    BD = torch.block_diag(*([H] * num_heads))
    return BD, is_hadamard


def phase_c(vision_model: nn.Module) -> dict:
    """R2: rotate per-head value/attention-output space by a block-diagonal H.

    Folded offline: V output rotated by BD (v_proj rows <- BD^T W, b <- BD^T b),
    out_proj input rotated by BD (out_proj cols <- W BD). Output-invariant because
    out_h @ H cancels (W_o_h @ H)^T through the projection; softmax is untouched
    (it does not depend on V). No runtime op -> v68-safe (unlike R3/R4 MLP online
    Hadamard, which cannot fold through the GELU and would need a runtime op).

    Acts on the head_dim axis (v_proj output / out_proj input); the residual-stream
    rotation Q (phase_b) acts on the embed_dim axis (v_proj input / out_proj
    output). Different axes -> the two folds compose cleanly. Encoder layers only;
    the head pooling MHA is left for a later R2 pass.
    """
    layers = vision_model.encoder.layers
    attn0 = layers[0].self_attn
    num_heads, head_dim = int(attn0.num_heads), int(attn0.head_dim)
    BD, is_hadamard = _blockdiag_headdim_rotation(num_heads, head_dim)
    BDt = BD.t().contiguous()

    for layer in layers:
        # V <- V @ BD : rotate the value output on head_dim (left-fold BD^T).
        fold_left_into_linear_writer(layer.self_attn.v_proj, BDt)
        # out_proj input <- @ BD : undo the rotation on the attention output.
        fold_right_into_reader(layer.self_attn.out_proj, BD)

    err = (BD @ BDt - torch.eye(BD.shape[0], dtype=torch.float64)).abs().max().item()
    return {
        "applied": True,
        "n_layers": len(layers),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hadamard": is_hadamard,
        "orthogonality_max_err": err,
    }


# --------------------------------------------------------------------------- #
# Gate: encode_image cosine vs reference
# --------------------------------------------------------------------------- #
def _embed(model: nn.Module, raw_paths: list[Path], image_size: int, device: str) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for raw_path in raw_paths:
            image = _read_raw_tensor(raw_path, image_size).to(device)
            out = model.encode_image(image).detach().float().cpu().reshape(-1)
            rows.append(out)
    return torch.stack(rows)


def _cosine_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    an = torch.nn.functional.normalize(a, dim=1)
    bn = torch.nn.functional.normalize(b, dim=1)
    cos = (an * bn).sum(dim=1)
    return {"mean": float(cos.mean()), "min": float(cos.min()), "max": float(cos.max())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotation-based equalization of the vision encoder (M1).")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_rotated")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10")
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--precision", choices=["fp32"], default="fp32", help="Rotation math requires fp32.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2400, help="Seed for the orthogonal Q.")
    p.add_argument("--gate-threshold", type=float, default=0.9999)
    p.add_argument("--skip-phase-b", action="store_true", help="Apply only Phase A (debug the LN->RMSNorm fold).")
    p.add_argument("--skip-r2", action="store_true", help="Skip Phase C (R2 head_dim Hadamard); reproduce the pre-R2 rotated model.")
    p.add_argument("--no-save", action="store_true", help="Run the gate but do not write the rotated model.")
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_rotated/rotation_summary.json",
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

    raw_paths = _parse_input_list(input_dir, args.input_list)
    print(f"Loading PyTorch model from {model_dir} ...")
    model, config = _load_pytorch_model(model_dir, args.precision, args.device)
    model.eval()

    vision_model = model.vision_model
    dim = vision_model.post_layernorm.normalized_shape[0]
    print(f"Vision residual dim = {dim}; encoder layers = {len(vision_model.encoder.layers)}")

    print(f"Computing reference embeddings on {len(raw_paths)} inputs ...")
    ref_emb = _embed(model, raw_paths, args.image_size, args.device)

    print("Phase A: fold LN affine into readers + set LN identity (keep fused LayerNorm) ...")
    a_info = phase_a(vision_model, dim)
    after_a = _embed(model, raw_paths, args.image_size, args.device)
    cos_a = _cosine_stats(ref_emb, after_a)
    print(f"  Phase A invariance cosine: mean={cos_a['mean']:.8f} min={cos_a['min']:.8f}")

    b_info: dict = {"applied": False}
    cos_b = cos_a
    if not args.skip_phase_b:
        print(f"Phase B: rotate residual by mean-preserving Q (seed={args.seed}) ...")
        Q = _mean_preserving_orthogonal(dim, args.seed)
        ones = torch.ones(dim, dtype=torch.float64)
        mean_err = (Q @ ones - ones).abs().max().item()
        b_info = {"applied": True, "mean_preserve_max_err": mean_err, **phase_b(vision_model, Q)}
        print(f"  Q orthogonality max err: {b_info['orthogonality_max_err']:.2e}; Q@1=1 max err: {mean_err:.2e}")
        after_b = _embed(model, raw_paths, args.image_size, args.device)
        cos_b = _cosine_stats(ref_emb, after_b)
        print(f"  Phase B invariance cosine: mean={cos_b['mean']:.8f} min={cos_b['min']:.8f}")

    c_info: dict = {"applied": False}
    cos_c = cos_b
    if not args.skip_r2:
        print("Phase C (R2): rotate per-head value/output space by block-diag Hadamard ...")
        c_info = phase_c(vision_model)
        print(
            f"  R2: heads={c_info['num_heads']} head_dim={c_info['head_dim']} "
            f"hadamard={c_info['hadamard']} orth_err={c_info['orthogonality_max_err']:.2e}"
        )
        after_c = _embed(model, raw_paths, args.image_size, args.device)
        cos_c = _cosine_stats(ref_emb, after_c)
        print(f"  Phase C invariance cosine: mean={cos_c['mean']:.8f} min={cos_c['min']:.8f}")

    passed = cos_c["min"] >= args.gate_threshold
    summary = {
        "model_dir": str(model_dir),
        "input_dir": str(input_dir),
        "num_samples": len(raw_paths),
        "dim": dim,
        "seed": args.seed,
        "gate_threshold": args.gate_threshold,
        "phase_a": {**a_info, "cosine": cos_a},
        "phase_b": {**b_info, "cosine": cos_b},
        "phase_c": {**c_info, "cosine": cos_c},
        "gate_passed": passed,
    }

    if passed and not args.no_save:
        out_dir = args.output_dir.expanduser().resolve()
        print(f"GATE PASS -> saving rotated model to {out_dir}")
        _save_rotated(model, model_dir, out_dir)
        summary["output_dir"] = str(out_dir)

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== ROTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if not passed:
        print(
            "\nGATE FAIL: encode_image cosine below threshold. Likely fold bug "
            "(mean-subtraction fold, affine fold order, or Q vs Q^T). Model NOT saved."
        )
        raise SystemExit(1)
    print("\nGATE PASS: FP32 output-invariant. Ready for M2 (export opset-20 rotated).")


if __name__ == "__main__":
    main()

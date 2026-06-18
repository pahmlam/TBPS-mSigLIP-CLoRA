#!/usr/bin/env python3
"""Learned mean-preserving rotation (SpinQuant-inspired) for the vision encoder.

Replaces the *random* mean-preserving orthogonal Q of `rotate_vision_encoder.py`
with a **learned** one. Coverage-based QAT (v1-v6) plateaued at QDQ cosine ~0.949
/ R@1 ~49.3 because that is the per-tensor W8A8 ceiling of a *random*-Q rotated
model. A learned Q lowers the ceiling by minimizing the actual quantization
difficulty of the rotated residual activations.

WHY THIS OBJECTIVE
------------------
Per-tensor INT8 error is set by the dynamic range `max_i|x_i|` (scale = max/127,
error variance ~ scale^2/12; §2A.3 of the method doc). An orthogonal Q preserves
energy but reshapes the distribution, so the deployable goal is to pick Q that
**minimizes the per-tensor max-abs** of every rotated residual activation that AI
Hub quantizes. We therefore cache those activations on calibration data and
minimize `Σ_tensor (max-abs of Q·a)^2` over a learnable mean-preserving Q. This is
the data-driven (learned) version of the incoherence reduction that a random
rotation does stochastically; it is fully offline and folds into weights exactly
like the random Q, so the rest of the pipeline (export → QAT → W8A8 → board) is
unchanged.

Mean-preserving constraint (Q·1 = 1, keeps fused LayerNorm) is enforced by
parametrizing Q = U · blockdiag(1, R_c) · U^T with U[:,0] = 1/sqrt(d) fixed and
R_c a learned orthogonal on the (d-1)-dim complement (Cayley transform of a
learnable skew-symmetric matrix).

GATE: after folding the learned Q, encode_image FP32 cosine vs the original model
must be ~1.0 (phase_a + phase_b are output-invariant by construction).

Example:
    venv/bin/python deployment/scripts/qnn/learn_rotation.py \
      --model-dir artifacts/deployment/exports/exported_model \
      --output-dir artifacts/deployment/exports/exported_model_rotated_learned \
      --calib-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
      --gate-input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
      --num-calib 128 --tokens-per-image 16 --steps 1500
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

from compare_qnn_with_pytorch import (  # noqa: E402
    _load_pytorch_model,
    _parse_input_list,
    _read_raw_tensor,
)
from rotate_vision_encoder import (  # noqa: E402
    _cosine_stats,
    _embed,
    phase_a,
    phase_b,
)


# --------------------------------------------------------------------------- #
# Cache the residual-stream activations that the (folded) rotation will quantize.
# Readers see LN outputs (-> Q·LN); writers emit out_proj/fc2 (-> Q·writer). We
# minimize the per-tensor max-abs of Q applied to all of these.
# --------------------------------------------------------------------------- #
def _collect_rotation_site_activations(
    vision_model: nn.Module,
    raw_paths: list[Path],
    image_size: int,
    device: str,
    tokens_per_image: int,
    seed: int,
) -> list[torch.Tensor]:
    """Return a list of (rows, dim) tensors, one per rotation-affected site.

    Sites: each layer_norm1 / layer_norm2 output (reader inputs), each
    self_attn.out_proj / mlp.fc2 output (writer outputs), and post_layernorm.
    Tokens are subsampled per image to bound memory.
    """
    layers = vision_model.encoder.layers
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
    handles.append(vision_model.post_layernorm.register_forward_hook(_hook("post_ln")))

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for raw_path in raw_paths:
            image = _read_raw_tensor(raw_path, image_size).to(device)
            vision_model(image)
            # subsample tokens from each freshly-appended capture
            for name, lst in captured.items():
                t = lst[-1]
                if tokens_per_image and t.shape[0] > tokens_per_image:
                    idx = torch.randperm(t.shape[0], generator=gen)[:tokens_per_image]
                    lst[-1] = t[idx].contiguous()

    for h in handles:
        h.remove()

    return [torch.cat(lst, dim=0) for lst in captured.values()]


# --------------------------------------------------------------------------- #
# Mean-preserving orthogonal Q parametrized by a learnable skew matrix (Cayley).
# --------------------------------------------------------------------------- #
class MeanPreservingRotation(nn.Module):
    def __init__(self, dim: int, seed: int, device: str) -> None:
        super().__init__()
        self.dim = dim
        g = torch.Generator().manual_seed(seed)
        v0 = torch.ones(dim, dtype=torch.float64) / (dim ** 0.5)
        A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        A[:, 0] = v0
        U, _ = torch.linalg.qr(A)
        if torch.dot(U[:, 0], v0) < 0:
            U[:, 0] = -U[:, 0]
        # U[:,0] spans the mean axis; columns 1.. span the complement.
        self.register_buffer("U", U.to(device))
        # Learnable skew on the (dim-1) complement; start at 0 -> Q = identity.
        self.skew_param = nn.Parameter(torch.zeros(dim - 1, dim - 1, dtype=torch.float64, device=device))

    def matrix(self) -> torch.Tensor:
        d = self.dim
        A = self.skew_param - self.skew_param.t()  # skew-symmetric
        eye = torch.eye(d - 1, dtype=A.dtype, device=A.device)
        # Cayley: R_c = (I - A)(I + A)^{-1}  (orthogonal)
        Rc = torch.linalg.solve((eye + A).t(), (eye - A).t()).t()
        R = torch.eye(d, dtype=A.dtype, device=A.device)
        R[1:, 1:] = Rc
        return self.U @ R @ self.U.t()


def _per_tensor_max_sq(rotated: torch.Tensor) -> torch.Tensor:
    """(max_i |x_i|)^2 over the whole tensor — the per-tensor INT8 scale driver."""
    return rotated.abs().max() ** 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn a mean-preserving rotation (SpinQuant-style).")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_rotated_learned")
    p.add_argument("--calib-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_train_calib_2000")
    p.add_argument("--gate-input-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10")
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--num-calib", type=int, default=128, help="Calibration images for caching activations.")
    p.add_argument("--tokens-per-image", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2400)
    p.add_argument("--gate-threshold", type=float, default=0.9999)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--json", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_rotated_learned/learned_rotation_summary.json")
    return p.parse_args()


def _save_rotated(model: nn.Module, src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "config.yaml", out_dir / "config.yaml")
    torch.save(model.state_dict(), out_dir / "model_fp32.pt")


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    device = args.device

    print(f"Loading model from {model_dir} ...")
    model, _ = _load_pytorch_model(model_dir, "fp32", device)
    model.eval()
    vision_model = model.vision_model
    dim = vision_model.post_layernorm.normalized_shape[0]

    gate_paths = _parse_input_list(args.gate_input_dir.expanduser().resolve(), args.input_list)
    print("Computing reference embeddings (pre-rotation) ...")
    ref_emb = _embed(model, gate_paths, args.image_size, device)

    print("Phase A: fold LN affine into readers, set LN identity ...")
    phase_a(vision_model, dim)

    calib_paths = _parse_input_list(args.calib_dir.expanduser().resolve(), args.input_list)
    if args.num_calib > 0:
        calib_paths = calib_paths[: args.num_calib]
    print(f"Caching rotation-site activations on {len(calib_paths)} calib images ...")
    acts = _collect_rotation_site_activations(
        vision_model, calib_paths, args.image_size, device,
        args.tokens_per_image, args.seed,
    )
    acts = [a.to(torch.float64).to(device) for a in acts]
    print(f"  cached {len(acts)} site tensors; rows each ~{acts[0].shape[0]}")

    # Baseline (identity rotation) objective for reference.
    with torch.no_grad():
        base_obj = sum(_per_tensor_max_sq(a) for a in acts).item()
        base_maxabs = max(a.abs().max().item() for a in acts)

    rot = MeanPreservingRotation(dim, args.seed, device)
    opt = torch.optim.Adam(rot.parameters(), lr=args.lr)

    print(f"Optimizing learned rotation: {args.steps} steps (base obj={base_obj:.1f}, max|a|={base_maxabs:.1f}) ...")
    for step in range(args.steps):
        opt.zero_grad()
        Q = rot.matrix()
        Qt = Q.t()
        obj = 0.0
        for a in acts:
            obj = obj + _per_tensor_max_sq(a @ Qt)  # rotated reader/writer activation = a · Q^T
        obj.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                cur_max = max((a @ rot.matrix().t()).abs().max().item() for a in acts)
            print(f"  step {step:4d}  obj={float(obj):.1f}  max|Qa|={cur_max:.2f}")

    with torch.no_grad():
        Q = rot.matrix().detach()
        orth_err = (Q @ Q.t() - torch.eye(dim, dtype=Q.dtype, device=device)).abs().max().item()
        ones = torch.ones(dim, dtype=Q.dtype, device=device)
        mean_err = (Q @ ones - ones).abs().max().item()
        final_obj = sum(_per_tensor_max_sq(a @ Q.t()) for a in acts).item()
        final_maxabs = max((a @ Q.t()).abs().max().item() for a in acts)

    print(f"Learned Q: orth_err={orth_err:.2e} mean_err={mean_err:.2e}")
    print(f"Objective {base_obj:.1f} -> {final_obj:.1f}  | max|a| {base_maxabs:.2f} -> {final_maxabs:.2f}")

    print("Phase B: fold learned Q into weights ...")
    phase_b(vision_model, Q.to(torch.float64))

    after = _embed(model, gate_paths, args.image_size, device)
    cos = _cosine_stats(ref_emb, after)
    print(f"Output-invariance cosine: mean={cos['mean']:.8f} min={cos['min']:.8f}")
    passed = cos["min"] >= args.gate_threshold

    summary = {
        "model_dir": str(model_dir),
        "dim": dim,
        "num_calib": len(calib_paths),
        "tokens_per_image": args.tokens_per_image,
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
        print(f"GATE PASS -> saving learned-rotated model to {out_dir}")
        _save_rotated(model, model_dir, out_dir)
        summary["output_dir"] = str(out_dir)

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== LEARNED ROTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("GATE FAIL: output not invariant after folding learned Q (fold/parametrization bug).")
    print("\nGATE PASS. Next: export_rotated_vision_onnx.py on the output dir, then QAT/quantize.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P1-A - SmoothQuant equalization for the vision encoder (output-invariant).

Plan: `deployment/docs/journal/[deploy-plan]-2026-06-14.md`, branch P1-A.

P0 found that INT8/A16 fidelity collapses because activation quantization is
*per-tensor* while a few fixed channels carry large values (concentration up to
~876x in the residual stream). SmoothQuant removes that per-tensor pressure by a
mathematically-invariant reparameterization:

For Y = X @ W (X = LayerNorm output, W shape (C_in, C_out)) we pick a per-input-
channel scale s and rewrite

    X' = X / s         (folded into the LayerNorm affine: gamma_j,beta_j /= s_j)
    W' = diag(s) @ W   (folded into the projection weight: W[j,:] *= s_j)

so X'@W' == X@W exactly in FP32, but the LayerNorm-output activation range is
flattened, which the downstream per-tensor INT8/A16 quantizer can represent.

    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

This produces an all-FP32 ONNX (no QDQ, no float-bypass) ready to push through
the *existing* AI Hub quantize pipeline. It is only applied where a LayerNorm
output feeds ONLY MatMul/Gemm projections with initializer weights, so invariance
is guaranteed; any other wiring is skipped.

Example:
    venv/bin/python deployment/scripts/qnn/smoothquant_equalize.py \\
      --onnx-model artifacts/deployment/exports/exported_model/vision_onnx \\
      --calib-input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \\
      --output-dir artifacts/deployment/exports/exported_model_smoothquant/vision_onnx \\
      --validate-input-dir artifacts/deployment/qnn_inputs/vn3k_test_10
"""

from __future__ import annotations

import argparse
import json
import re
from array import array
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LN_WEIGHT_RE = re.compile(r"encoder\.layers\.(\d+)\.(layer_norm1|layer_norm2)\.weight")


# --------------------------------------------------------------------------- #
# Inputs (raw float32, same as compare/qnn-net-run)
# --------------------------------------------------------------------------- #
def _parse_input_list(input_dir: Path, input_list_name: str) -> list[Path]:
    input_list = input_dir / input_list_name
    if not input_list.exists():
        raise FileNotFoundError(f"Input list not found: {input_list}")
    paths: list[Path] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        value = line.split(":=", 1)[1] if ":=" in line else line.split()[-1]
        p = Path(value)
        paths.append(p if p.is_absolute() else input_dir / p)
    if not paths:
        raise ValueError(f"No raw inputs in {input_list}")
    return paths


def _read_raw_image(path: Path, image_size: int) -> np.ndarray:
    expected = 3 * image_size * image_size
    data = path.read_bytes()
    if len(data) != expected * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {expected * 4}")
    values = array("f")
    values.frombytes(data)
    import sys

    if sys.byteorder != "little":
        values.byteswap()
    return np.asarray(values, dtype=np.float32).reshape(1, 3, image_size, image_size)


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path
    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx in {path}, found: {candidates}")
    return candidates[0]


# --------------------------------------------------------------------------- #
# Structure discovery
# --------------------------------------------------------------------------- #
def _consumers_by_input(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            out.setdefault(name, []).append(node)
    return out


def _find_ln_nodes(graph: onnx.GraphProto) -> dict[tuple[int, str], onnx.NodeProto]:
    """Map (layer_index, 'layer_norm1'|'layer_norm2') -> LayerNormalization node."""
    found: dict[tuple[int, str], onnx.NodeProto] = {}
    for node in graph.node:
        if node.op_type != "LayerNormalization":
            continue
        for name in node.input:
            m = LN_WEIGHT_RE.search(name)
            if m:
                found[(int(m.group(1)), m.group(2))] = node
    return found


def _ln_affine_names(node: onnx.NodeProto) -> tuple[str, str]:
    """Return (weight_name, bias_name) of a LayerNormalization node."""
    weight = node.input[1] if len(node.input) > 1 else ""
    bias = node.input[2] if len(node.input) > 2 else ""
    return weight, bias


class _SmoothSite:
    """One foldable LayerNorm -> {projection weights} site."""

    def __init__(self, key, ln_node, gamma, beta, out_tensor, weight_names):
        self.key = key
        self.ln_node = ln_node
        self.gamma = gamma
        self.beta = beta
        self.out_tensor = out_tensor
        self.weight_names = weight_names


def _discover_sites(
    graph: onnx.GraphProto, blocks: set[int] | None
) -> tuple[list[_SmoothSite], list[dict]]:
    consumers = _consumers_by_input(graph)
    initializers = {init.name for init in graph.initializer}
    ln_nodes = _find_ln_nodes(graph)

    sites: list[_SmoothSite] = []
    skipped: list[dict] = []
    for (layer, which), ln_node in sorted(ln_nodes.items()):
        if blocks is not None and layer not in blocks:
            continue
        out_tensor = ln_node.output[0]
        gamma, beta = _ln_affine_names(ln_node)
        if gamma not in initializers or beta not in initializers:
            skipped.append({"layer": layer, "which": which, "reason": "affine not initializer"})
            continue

        out_consumers = consumers.get(out_tensor, [])
        weight_names: list[str] = []
        foldable = bool(out_consumers)
        for c in out_consumers:
            if c.op_type not in {"MatMul", "Gemm"}:
                foldable = False
                break
            # activation is input[0]; weight is the initializer input (input[1])
            w = next((inp for inp in c.input if inp in initializers), None)
            if w is None:
                foldable = False
                break
            weight_names.append(w)
        if not foldable:
            skipped.append(
                {"layer": layer, "which": which, "reason": "LN output has non-foldable consumer"}
            )
            continue

        sites.append(
            _SmoothSite((layer, which), ln_node, gamma, beta, out_tensor, weight_names)
        )
    return sites, skipped


# --------------------------------------------------------------------------- #
# Calibration: per-channel abs-max of each LN output
# --------------------------------------------------------------------------- #
def _calibrate_ln_outputs(
    model: onnx.ModelProto,
    tensor_names: list[str],
    raw_paths: list[Path],
    image_size: int,
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    cal_model = onnx.ModelProto()
    cal_model.CopyFrom(model)
    existing = {o.name for o in cal_model.graph.output}
    for name in tensor_names:
        if name not in existing:
            vi = onnx.ValueInfoProto()
            vi.name = name
            cal_model.graph.output.append(vi)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        cal_model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    acc: dict[str, np.ndarray] = {}
    for raw_path in raw_paths:
        image = _read_raw_image(raw_path, image_size)
        outs = sess.run(tensor_names, {input_name: image})
        for name, val in zip(tensor_names, outs):
            v = np.abs(np.asarray(val, dtype=np.float32))
            per_ch = v.reshape(-1, v.shape[-1]).max(axis=0)
            acc[name] = per_ch if name not in acc else np.maximum(acc[name], per_ch)
    return acc


# --------------------------------------------------------------------------- #
# Embedding inference (for invariance validation)
# --------------------------------------------------------------------------- #
def _embed(model: onnx.ModelProto, raw_paths: list[Path], image_size: int) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    rows = []
    for raw_path in raw_paths:
        image = _read_raw_image(raw_path, image_size)
        rows.append(np.asarray(sess.run([out_name], {input_name: image})[0]).reshape(-1))
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# Fold
# --------------------------------------------------------------------------- #
def _apply_smoothquant(
    model: onnx.ModelProto,
    sites: list[_SmoothSite],
    activation_max: dict[str, np.ndarray],
    alpha: float,
    clip_min: float,
    clip_max: float,
    eps: float,
) -> list[dict]:
    init_by_name = {init.name: init for init in model.graph.initializer}
    report: list[dict] = []

    for site in sites:
        a = activation_max[site.out_tensor].astype(np.float64)
        weights = {name: numpy_helper.to_array(init_by_name[name]).astype(np.float64) for name in site.weight_names}
        # per-input-channel (axis 0) weight abs-max across all consumer projections
        w = None
        for arr in weights.values():
            row_max = np.abs(arr).max(axis=1)  # (C_in,)
            w = row_max if w is None else np.maximum(w, row_max)

        s = np.power(np.maximum(a, eps), alpha) / np.power(np.maximum(w, eps), 1.0 - alpha)
        # leave dead channels untouched; clip to avoid creating new outliers
        s = np.where((a < eps) | (w < eps), 1.0, s)
        s = np.clip(s, clip_min, clip_max)

        # Fold into LayerNorm affine: gamma,beta /= s
        for affine_name in (site.gamma, site.beta):
            init = init_by_name[affine_name]
            arr = numpy_helper.to_array(init).astype(np.float64)
            new = (arr / s).astype(numpy_helper.to_array(init).dtype)
            init.CopyFrom(numpy_helper.from_array(new, affine_name))
        # Fold into each projection weight: W[j,:] *= s_j  (axis 0 = input channel)
        for name, arr in weights.items():
            init = init_by_name[name]
            new = (arr * s[:, None]).astype(numpy_helper.to_array(init).dtype)
            init.CopyFrom(numpy_helper.from_array(new, name))

        report.append(
            {
                "layer": site.key[0],
                "which": site.key[1],
                "ln_output": site.out_tensor,
                "num_proj_weights": len(site.weight_names),
                "channels": int(s.shape[0]),
                "act_absmax_before": float(a.max()),
                "act_absmax_after_est": float((a / s).max()),
                "scale_min": float(s.min()),
                "scale_max": float(s.max()),
            }
        )
    return report


def _save_external(model: onnx.ModelProto, output_onnx: Path) -> None:
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    data_name = output_onnx.name + ".data"
    data_path = output_onnx.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(output_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmoothQuant equalization for the vision encoder.")
    p.add_argument(
        "--onnx-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model/vision_onnx",
    )
    p.add_argument(
        "--calib-input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10",
        help="Raw inputs used to estimate per-channel activation max. More is better.",
    )
    p.add_argument(
        "--validate-input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10",
        help="Raw inputs used to verify FP32 output invariance (held vs original).",
    )
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--blocks", default="all", help="'all', '4-11', or '4,5,6'.")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--clip-min", type=float, default=1e-2)
    p.add_argument("--clip-max", type=float, default=1e2)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/exports/exported_model_smoothquant/vision_onnx",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/exports/exported_model_smoothquant/smoothquant_summary.json",
        help="Summary path; keep OUTSIDE the vision_onnx model dir (AI Hub rejects non-onnx/.data files there).",
    )
    return p.parse_args()


def _parse_blocks(spec: str) -> set[int] | None:
    if spec.strip().lower() == "all":
        return None
    blocks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            blocks.update(range(int(lo), int(hi) + 1))
        else:
            blocks.add(int(part))
    return blocks


def main() -> None:
    args = parse_args()
    onnx_path = _find_single_onnx(args.onnx_model.expanduser().resolve())
    blocks = _parse_blocks(args.blocks)

    print(f"Loading FP32 ONNX (with weights): {onnx_path}")
    model = onnx.load(str(onnx_path), load_external_data=True)
    sites, skipped = _discover_sites(model.graph, blocks)
    if not sites:
        raise SystemExit(f"No foldable LayerNorm->projection sites found (skipped={skipped}).")
    print(f"Foldable sites: {len(sites)} | skipped: {len(skipped)}")

    # Original embeddings BEFORE editing (for invariance check).
    val_paths = _parse_input_list(args.validate_input_dir.expanduser().resolve(), args.input_list)
    print(f"Computing original embeddings on {len(val_paths)} validation inputs...")
    orig_emb = _embed(model, val_paths, args.image_size)

    # Calibrate activation per-channel max at each LN output.
    calib_paths = _parse_input_list(args.calib_input_dir.expanduser().resolve(), args.input_list)
    print(f"Calibrating activation ranges on {len(calib_paths)} inputs...")
    ln_outputs = [s.out_tensor for s in sites]
    act_max = _calibrate_ln_outputs(model, ln_outputs, calib_paths, args.image_size)

    print("Applying SmoothQuant fold...")
    report = _apply_smoothquant(
        model, sites, act_max, args.alpha, args.clip_min, args.clip_max, args.eps
    )

    # Invariance check: equalized embeddings vs original.
    print("Verifying FP32 output invariance...")
    new_emb = _embed(model, val_paths, args.image_size)
    cosines = []
    for a, b in zip(orig_emb, new_emb):
        an, bn = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
        cosines.append(float(np.dot(an, bn)))
    cos_mean, cos_min = float(np.mean(cosines)), float(np.min(cosines))

    output_onnx = args.output_dir.expanduser().resolve() / onnx_path.name
    _save_external(model, output_onnx)

    act_before = max(r["act_absmax_before"] for r in report)
    act_after = max(r["act_absmax_after_est"] for r in report)
    summary = {
        "source_onnx": str(onnx_path),
        "output_onnx": str(output_onnx),
        "blocks": "all" if blocks is None else sorted(blocks),
        "alpha": args.alpha,
        "clip": [args.clip_min, args.clip_max],
        "num_sites": len(sites),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "calib_samples": len(calib_paths),
        "validate_samples": len(val_paths),
        "invariance_cosine_mean": cos_mean,
        "invariance_cosine_min": cos_min,
        "max_ln_act_absmax_before": act_before,
        "max_ln_act_absmax_after_est": act_after,
        "act_range_reduction_x": act_before / max(act_after, 1e-12),
        "per_site": report,
    }
    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== SMOOTHQUANT SUMMARY ===")
    print(f"  sites folded:                {len(sites)} (skipped {len(skipped)})")
    print(f"  LN-output act abs-max:       {act_before:.1f} -> {act_after:.1f}  ({act_before/max(act_after,1e-12):.1f}x reduction)")
    print(f"  FP32 invariance cosine:      mean={cos_mean:.6f}  min={cos_min:.6f}")
    print(f"  equalized model:             {output_onnx}")
    print(f"  summary:                     {args.json}")
    if cos_min < 0.999:
        print("  WARNING: invariance below 0.999 — fold may be incorrect; do NOT quantize this yet.")
    else:
        print("  OK: output-invariant. Ready to push through AI Hub quantize.")


if __name__ == "__main__":
    main()

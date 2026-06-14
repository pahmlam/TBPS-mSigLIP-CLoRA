#!/usr/bin/env python3
"""P0.1 - Profile activation outliers in vision-encoder blocks 4-11.

This is a pre-flight diagnostic for the activation-outlier-aware QDQ plan
(`deployment/docs/journal/[deploy-plan]-2026-06-14.md`). It runs on the FP32
vision ONNX (no QDQ) and characterises *how* the mid-to-late encoder blocks
carry large activations, so we can pick the right deployable fix:

- Outliers concentrated in a few *fixed* channels  -> SmoothQuant / per-channel
  equalization can flatten them into an all-INT8 graph (plan branch P1-A).
- Outliers diffuse / input-dependent                -> per-tensor INT8 cannot
  represent them; 16-bit activations are likely required (plan branch P1-C).

It is dependency-light on purpose: `onnx` + `onnxruntime` + `numpy` only, and it
reads the same raw float32 inputs that `qnn-net-run` and the compare scripts use,
so the measured ranges are the encoder's, not preprocessing drift.

Example:
    venv/bin/python deployment/scripts/qnn/profile_activation_outliers.py \\
      --onnx-model artifacts/deployment/exports/exported_model/vision_onnx \\
      --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \\
      --blocks 4-11 \\
      --json artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/summary.json \\
      --csv  artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/per_tensor.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from array import array
from pathlib import Path

import numpy as np
import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default ops whose activation tensors we inspect. Add/LayerNormalization track
# the residual stream and normalized inputs (where ViT "massive activations"
# live); MatMul/Gemm activation inputs are the literal per-tensor activation
# quantization sites.
DEFAULT_CAPTURE_OPS = ("Add", "LayerNormalization", "MatMul", "Gemm")


# --------------------------------------------------------------------------- #
# Graph / block structure (mirrors qdq_surgery.py block identification)
# --------------------------------------------------------------------------- #
def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path
    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx file in {path}, found: {candidates}")
    return candidates[0]


def _encoder_block_start_indices(graph: onnx.GraphProto) -> dict[int, int]:
    """Map encoder block index -> node index of its layer_norm1."""
    starts: dict[int, int] = {}
    pattern = re.compile(r"encoder\.layers\.(\d+)\.layer_norm1\.weight")
    for index, node in enumerate(graph.node):
        if node.op_type != "LayerNormalization":
            continue
        for input_name in node.input:
            match = pattern.search(input_name)
            if match:
                starts[int(match.group(1))] = index
    return starts


def _post_layernorm_start_index(graph: onnx.GraphProto) -> int | None:
    for index, node in enumerate(graph.node):
        if node.op_type != "LayerNormalization":
            continue
        if any("vision_model.post_layernorm.weight" in name for name in node.input):
            return index
    return None


def _block_node_ranges(graph: onnx.GraphProto, blocks: set[int]) -> dict[int, tuple[int, int]]:
    """Return {block: (start_index, end_index)} half-open node ranges."""
    starts = _encoder_block_start_indices(graph)
    post_start = _post_layernorm_start_index(graph)
    ranges: dict[int, tuple[int, int]] = {}
    for block in sorted(blocks):
        if block not in starts:
            raise ValueError(
                f"Could not find encoder block {block} in ONNX graph "
                f"(found blocks: {sorted(starts)})"
            )
        later = [idx for num, idx in starts.items() if num > block]
        end = min(later) if later else (post_start if post_start is not None else len(graph.node))
        ranges[block] = (starts[block], end)
    return ranges


def _block_of_node_index(node_index: int, ranges: dict[int, tuple[int, int]]) -> int | None:
    for block, (start, end) in ranges.items():
        if start <= node_index < end:
            return block
    return None


# --------------------------------------------------------------------------- #
# Target tensor selection
# --------------------------------------------------------------------------- #
def _select_target_tensors(
    graph: onnx.GraphProto,
    ranges: dict[int, tuple[int, int]],
    capture_ops: set[str],
) -> list[dict]:
    """Pick activation tensors to observe within the block ranges.

    For Add / LayerNormalization we observe the node *output* (residual stream /
    normalized hidden state). For MatMul / Gemm we observe the *activation
    input* (non-initializer), i.e. the tensor that activation quantization would
    encode.
    """
    initializers = {init.name for init in graph.initializer}
    targets: dict[str, dict] = {}

    for node_index, node in enumerate(graph.node):
        block = _block_of_node_index(node_index, ranges)
        if block is None or node.op_type not in capture_ops:
            continue

        if node.op_type in {"Add", "LayerNormalization"}:
            for out in node.output:
                if out and out not in targets:
                    targets[out] = {
                        "tensor": out,
                        "block": block,
                        "producer_op": node.op_type,
                        "role": "output",
                    }
        elif node.op_type in {"MatMul", "Gemm"}:
            for in_name in node.input:
                if in_name and in_name not in initializers and in_name not in targets:
                    targets[in_name] = {
                        "tensor": in_name,
                        "block": block,
                        "producer_op": f"{node.op_type}_input",
                        "role": "activation_input",
                    }

    return list(targets.values())


def _add_graph_outputs(model: onnx.ModelProto, tensor_names: list[str]) -> list[str]:
    """Expose internal tensors as graph outputs so ORT returns them.

    Bare ValueInfoProto (name only) is used; ORT resolves type/shape at runtime.
    Returns the list of names actually appended (skips existing outputs).
    """
    existing = {out.name for out in model.graph.output}
    added: list[str] = []
    for name in tensor_names:
        if name in existing:
            continue
        vi = onnx.ValueInfoProto()
        vi.name = name
        model.graph.output.append(vi)
        existing.add(name)
        added.append(name)
    return added


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _parse_input_list(input_dir: Path, input_list_name: str) -> list[Path]:
    input_list = input_dir / input_list_name
    if not input_list.exists():
        raise FileNotFoundError(f"Input list not found: {input_list}")
    raw_paths: list[Path] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        if ":=" in line:
            _, value = line.split(":=", 1)
        else:
            parts = line.split()
            value = parts[-1] if parts else ""
        raw_path = Path(value)
        if not raw_path.is_absolute():
            raw_path = input_dir / raw_path
        raw_paths.append(raw_path)
    if not raw_paths:
        raise ValueError(f"No raw input paths found in {input_list}")
    return raw_paths


def _read_raw_image(path: Path, image_size: int) -> np.ndarray:
    expected_values = 3 * image_size * image_size
    data = path.read_bytes()
    if len(data) != expected_values * 4:
        raise ValueError(
            f"{path} has {len(data)} bytes; expected {expected_values * 4} "
            f"for 1x3x{image_size}x{image_size} float32"
        )
    values = array("f")
    values.frombytes(data)
    import sys

    if sys.byteorder != "little":
        values.byteswap()
    return np.asarray(values, dtype=np.float32).reshape(1, 3, image_size, image_size)


# --------------------------------------------------------------------------- #
# Per-tensor statistics
# --------------------------------------------------------------------------- #
def _per_channel_abs_max(values: np.ndarray) -> np.ndarray:
    """abs-max over every axis except the last (treated as channel/feature)."""
    abs_values = np.abs(values)
    if abs_values.ndim == 1:
        return abs_values
    return abs_values.reshape(-1, abs_values.shape[-1]).max(axis=0)


class _TensorAccumulator:
    """Aggregate per-channel and percentile stats for one tensor over samples."""

    def __init__(self, meta: dict, top_k: int) -> None:
        self.meta = meta
        self.top_k = top_k
        self.shape: tuple[int, ...] | None = None
        self.per_channel_max: np.ndarray | None = None
        self.global_abs_max = 0.0
        self.p999_per_sample: list[float] = []
        self.p9999_per_sample: list[float] = []
        self.top_channel_sets: list[set[int]] = []
        self.has_nonfinite = False

    def update(self, values: np.ndarray) -> None:
        if not np.all(np.isfinite(values)):
            self.has_nonfinite = True
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        self.shape = tuple(values.shape)
        pcam = _per_channel_abs_max(values)
        if self.per_channel_max is None:
            self.per_channel_max = pcam.copy()
        else:
            self.per_channel_max = np.maximum(self.per_channel_max, pcam)
        self.global_abs_max = max(self.global_abs_max, float(pcam.max()))
        flat_abs = np.abs(values).reshape(-1)
        self.p999_per_sample.append(float(np.percentile(flat_abs, 99.9)))
        self.p9999_per_sample.append(float(np.percentile(flat_abs, 99.99)))
        k = min(self.top_k, pcam.shape[0])
        top_idx = set(np.argsort(pcam)[-k:].tolist())
        self.top_channel_sets.append(top_idx)

    def summarize(self) -> dict:
        pcam = self.per_channel_max if self.per_channel_max is not None else np.zeros(1)
        median = float(np.median(pcam))
        eps = 1e-12
        concentration = float(pcam.max() / (median + eps))
        outlier_factor = 10.0
        num_outlier_channels = int(np.sum(pcam > outlier_factor * (median + eps)))
        order = np.argsort(pcam)[::-1]
        k = min(self.top_k, pcam.shape[0])
        top_channels = [
            {"channel": int(order[i]), "abs_max": float(pcam[order[i]])} for i in range(k)
        ]
        # Cross-sample stability of which channels are extreme.
        if self.top_channel_sets:
            inter = set.intersection(*self.top_channel_sets)
            union = set.union(*self.top_channel_sets)
            stability_intersection_frac = len(inter) / float(k)
            stability_union_size = len(union)
        else:
            stability_intersection_frac = 0.0
            stability_union_size = 0
        return {
            **self.meta,
            "shape": list(self.shape) if self.shape else [],
            "num_channels": int(pcam.shape[0]),
            "global_abs_max": self.global_abs_max,
            "per_channel_abs_max_median": median,
            "per_channel_abs_max_max": float(pcam.max()),
            "concentration_ratio": concentration,
            "num_outlier_channels_gt_10x_median": num_outlier_channels,
            "p99_9_mean": float(np.mean(self.p999_per_sample)),
            "p99_9_max": float(np.max(self.p999_per_sample)),
            "p99_99_mean": float(np.mean(self.p9999_per_sample)),
            "p99_99_max": float(np.max(self.p9999_per_sample)),
            "top_channels": top_channels,
            "top_k_stability_intersection_frac": stability_intersection_frac,
            "top_k_union_size": stability_union_size,
            "has_nonfinite": self.has_nonfinite,
        }


# --------------------------------------------------------------------------- #
# Verdict heuristic
# --------------------------------------------------------------------------- #
def _verdict(per_tensor: list[dict], concentration_threshold: float, stability_threshold: float) -> dict:
    if not per_tensor:
        return {"decision": "NO_TENSORS", "reason": "No target tensors captured."}

    # Focus on the worst tensors by global abs-max (the ones that set the scale).
    ranked = sorted(per_tensor, key=lambda r: r["global_abs_max"], reverse=True)
    worst = ranked[: max(1, len(ranked) // 4)]  # top quartile
    concentrated = [
        r for r in worst if r["concentration_ratio"] >= concentration_threshold
    ]
    fixed = [
        r
        for r in concentrated
        if r["top_k_stability_intersection_frac"] >= stability_threshold
    ]

    frac_concentrated = len(concentrated) / len(worst)
    frac_fixed = (len(fixed) / len(concentrated)) if concentrated else 0.0

    if frac_concentrated >= 0.5 and frac_fixed >= 0.5:
        decision = "CONCENTRATED_FIXED"
        recommend = (
            "Outliers sit in a few stable channels of the worst tensors. "
            "SmoothQuant / per-channel equalization (plan branch P1-A) is the "
            "first thing to try; it can flatten these into an all-INT8 graph."
        )
    elif frac_concentrated >= 0.5 and frac_fixed < 0.5:
        decision = "CONCENTRATED_SHIFTING"
        recommend = (
            "Outliers are concentrated but the dominant channels move across "
            "inputs. Per-channel weight equalization alone may be unstable; "
            "prefer per-channel activation if HTP allows, else 16-bit activations "
            "(plan branch P1-C)."
        )
    else:
        decision = "DIFFUSE"
        recommend = (
            "Outlier energy is spread across many channels. Per-tensor INT8 "
            "cannot represent it without collapse; properly-calibrated W8A16 "
            "(plan branch P1-C) is the likely deployable path."
        )

    return {
        "decision": decision,
        "recommendation": recommend,
        "worst_tensor_count": len(worst),
        "frac_worst_concentrated": frac_concentrated,
        "frac_concentrated_fixed": frac_fixed,
        "global_abs_max_overall": ranked[0]["global_abs_max"],
        "global_abs_max_tensor": ranked[0]["tensor"],
        "concentration_threshold": concentration_threshold,
        "stability_threshold": stability_threshold,
    }


def _per_block_summary(per_tensor: list[dict]) -> dict:
    blocks: dict[str, dict] = {}
    for row in per_tensor:
        key = str(row["block"])
        cur = blocks.setdefault(
            key,
            {"block": row["block"], "num_tensors": 0, "max_global_abs_max": 0.0,
             "max_concentration_ratio": 0.0},
        )
        cur["num_tensors"] += 1
        cur["max_global_abs_max"] = max(cur["max_global_abs_max"], row["global_abs_max"])
        cur["max_concentration_ratio"] = max(
            cur["max_concentration_ratio"], row["concentration_ratio"]
        )
    return blocks


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "tensor", "block", "producer_op", "role", "shape", "num_channels",
        "global_abs_max", "per_channel_abs_max_median", "per_channel_abs_max_max",
        "concentration_ratio", "num_outlier_channels_gt_10x_median",
        "p99_9_mean", "p99_99_mean", "p99_99_max",
        "top_k_stability_intersection_frac", "top_k_union_size", "has_nonfinite",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["shape"] = "x".join(str(d) for d in row.get("shape", []))
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile activation outliers in encoder blocks.")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model/vision_onnx",
        help="FP32 vision ONNX file or directory containing one .onnx file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10",
    )
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--blocks", default="4-11", help="e.g. '4-11' or '4,5,6'.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all inputs.")
    parser.add_argument("--top-k", type=int, default=8, help="Top channels tracked per tensor.")
    parser.add_argument(
        "--capture-ops",
        default=",".join(DEFAULT_CAPTURE_OPS),
        help="Comma-separated ONNX op types to observe.",
    )
    parser.add_argument("--concentration-threshold", type=float, default=50.0)
    parser.add_argument("--stability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/summary.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/per_tensor.csv",
    )
    return parser.parse_args()


def _parse_blocks(spec: str) -> set[int]:
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
    import onnxruntime as ort

    onnx_path = _find_single_onnx(args.onnx_model.expanduser().resolve())
    input_dir = args.input_dir.expanduser().resolve()
    blocks = _parse_blocks(args.blocks)
    capture_ops = {op.strip() for op in args.capture_ops.split(",") if op.strip()}

    print(f"Loading ONNX (with external weights): {onnx_path}")
    model = onnx.load(str(onnx_path), load_external_data=True)
    ranges = _block_node_ranges(model.graph, blocks)
    targets = _select_target_tensors(model.graph, ranges, capture_ops)
    if not targets:
        raise SystemExit("No target tensors found for the requested blocks/ops.")
    target_names = [t["tensor"] for t in targets]
    added = _add_graph_outputs(model, target_names)
    print(f"Blocks: {sorted(blocks)} | target tensors: {len(targets)} | exposed as outputs: {len(added)}")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    # Keep intermediate tensors alive (no fusion) so requested outputs exist.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    fetch = [t["tensor"] for t in targets]

    raw_paths = _parse_input_list(input_dir, args.input_list)
    if args.max_samples > 0:
        raw_paths = raw_paths[: args.max_samples]

    accumulators = {t["tensor"]: _TensorAccumulator(t, args.top_k) for t in targets}
    for idx, raw_path in enumerate(raw_paths):
        image = _read_raw_image(raw_path, args.image_size)
        outputs = session.run(fetch, {input_name: image})
        for name, value in zip(fetch, outputs):
            accumulators[name].update(np.asarray(value))
        print(f"  [{idx + 1}/{len(raw_paths)}] {raw_path.name}")

    per_tensor = [acc.summarize() for acc in accumulators.values()]
    per_tensor.sort(key=lambda r: r["global_abs_max"], reverse=True)
    verdict = _verdict(per_tensor, args.concentration_threshold, args.stability_threshold)
    per_block = _per_block_summary(per_tensor)

    summary = {
        "onnx_model": str(onnx_path),
        "input_dir": str(input_dir),
        "num_samples": len(raw_paths),
        "blocks": sorted(blocks),
        "capture_ops": sorted(capture_ops),
        "num_target_tensors": len(per_tensor),
        "verdict": verdict,
        "per_block": per_block,
        "top_tensors_by_abs_max": per_tensor[:15],
    }

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(per_tensor, args.csv.expanduser().resolve())

    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nFull summary: {args.json}")
    print(f"Per-tensor CSV: {args.csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T0 — Profile activation outliers in the TEXT encoder (decide if rotation is needed).

Text counterpart of `profile_activation_outliers.py`. Reuses that module's block
detection, per-tensor accumulator, and verdict heuristic; only the input feeding
differs (text takes two integer inputs, `input_ids` + `attention_mask`, from
`prepare_vn3k_text_inputs.py`).

Purpose: measure the residual-stream concentration of the text encoder. If it is
mild (low concentration), plain per-tensor W8A8 may suffice and we can SKIP the
mean-preserving rotation that the vision encoder required (concentration 252x). If
it is severe, replicate the rotation.

Example:
    venv/bin/python deployment/scripts/qnn/profile_text_activation_outliers.py \
      --onnx-model artifacts/deployment/exports/exported_model/text_onnx \
      --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 \
      --blocks 0-11 \
      --json artifacts/deployment/runtime/diag/text_outliers/summary.json \
      --csv  artifacts/deployment/runtime/diag/text_outliers/per_tensor.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from profile_activation_outliers import (  # noqa: E402
    DEFAULT_CAPTURE_OPS,
    _add_graph_outputs,
    _block_node_ranges,
    _parse_blocks,
    _per_block_summary,
    _select_target_tensors,
    _TensorAccumulator,
    _verdict,
    _write_csv,
)
from compare_text_onnx_with_pytorch import (  # noqa: E402
    _find_single_onnx,
    _parse_dual_input_list,
    _read_raw_ints,
    _NP_BY_ORT,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile activation outliers in the text encoder.")
    p.add_argument(
        "--onnx-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model/text_onnx",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10",
    )
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--blocks", default="0-11")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--capture-ops", default=",".join(DEFAULT_CAPTURE_OPS))
    p.add_argument("--concentration-threshold", type=float, default=50.0)
    p.add_argument("--stability-threshold", type=float, default=0.5)
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/runtime/diag/text_outliers/summary.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/runtime/diag/text_outliers/per_tensor.csv",
    )
    return p.parse_args()


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
    print(f"Blocks: {sorted(blocks)} | target tensors: {len(targets)} | exposed: {len(added)}")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    in_types = {i.name: i.type for i in session.get_inputs()}
    fetch = [t["tensor"] for t in targets]

    rows_in = _parse_dual_input_list(input_dir, args.input_list)
    if args.max_samples > 0:
        rows_in = rows_in[: args.max_samples]

    accumulators = {t["tensor"]: _TensorAccumulator(t, args.top_k) for t in targets}
    for idx, entry in enumerate(rows_in):
        ids = _read_raw_ints(entry["input_ids"], args.seq_len)
        attn = _read_raw_ints(entry["attention_mask"], args.seq_len)
        feed = {
            "input_ids": ids.astype(_NP_BY_ORT.get(in_types.get("input_ids"), np.int64)),
            "attention_mask": attn.astype(_NP_BY_ORT.get(in_types.get("attention_mask"), np.int64)),
        }
        outputs = session.run(fetch, feed)
        for name, value in zip(fetch, outputs):
            accumulators[name].update(np.asarray(value))
        print(f"  [{idx + 1}/{len(rows_in)}]")

    per_tensor = [acc.summarize() for acc in accumulators.values()]
    per_tensor.sort(key=lambda r: r["global_abs_max"], reverse=True)
    verdict = _verdict(per_tensor, args.concentration_threshold, args.stability_threshold)
    per_block = _per_block_summary(per_tensor)

    summary = {
        "onnx_model": str(onnx_path),
        "input_dir": str(input_dir),
        "num_samples": len(rows_in),
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

    print("\n=== TEXT VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nFull summary: {args.json}")


if __name__ == "__main__":
    main()

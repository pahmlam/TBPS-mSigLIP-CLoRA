#!/usr/bin/env python3
"""Inspect QDQ scales on text self-attention masked logits.

The text W8A8 failure mode is a QuantizeLinear -> DequantizeLinear pair on
`self_attn/Add_output_0` (the tensor after `scores + mask`) before Softmax. If
the mask is -FLT_MAX, the scale becomes ~1e32. A finite mask should keep these
scales small enough for normal attention logits.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


LAYER_RE = re.compile(r"/text_model/encoder/layers\.(\d+)/self_attn/Softmax")


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path

    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx file in {path}, found {candidates}")
    return candidates[0]


def _producer_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output}


def _initializer_array(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray | None:
    tensor = initializers.get(name)
    if tensor is None or tensor.data_location == TensorProto.EXTERNAL:
        return None
    return numpy_helper.to_array(tensor)


def _scalar(value: np.ndarray | None) -> float | int | None:
    if value is None:
        return None
    if value.size != 1:
        return None
    item = value.reshape(-1)[0]
    if np.issubdtype(value.dtype, np.integer):
        return int(item)
    return float(item)


def _find_softmax_input_qdq(
    graph: onnx.GraphProto,
) -> list[dict[str, object]]:
    producer = _producer_by_output(graph)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    rows: list[dict[str, object]] = []

    for node in graph.node:
        match = LAYER_RE.search(node.name)
        if node.op_type != "Softmax" or match is None or not node.input:
            continue

        layer = int(match.group(1))
        softmax_input = node.input[0]
        dq_node = producer.get(softmax_input)
        q_node = producer.get(dq_node.input[0]) if dq_node and dq_node.input else None
        found = (
            dq_node is not None
            and dq_node.op_type == "DequantizeLinear"
            and q_node is not None
            and q_node.op_type == "QuantizeLinear"
        )

        row: dict[str, object] = {
            "layer": layer,
            "softmax_node": node.name,
            "softmax_input": softmax_input,
            "found_qdq": found,
        }
        if not found:
            rows.append(row)
            continue

        float_tensor = q_node.input[0] if q_node.input else ""
        scale_name = q_node.input[1] if len(q_node.input) > 1 else ""
        zero_point_name = q_node.input[2] if len(q_node.input) > 2 else ""
        scale = _initializer_array(initializers, scale_name)
        zero_point = _initializer_array(initializers, zero_point_name)

        row.update(
            {
                "float_tensor": float_tensor,
                "q_node": q_node.name,
                "dq_node": dq_node.name,
                "scale_name": scale_name,
                "scale": _scalar(scale),
                "zero_point_name": zero_point_name,
                "zero_point": _scalar(zero_point),
                "zero_point_dtype": "" if zero_point is None else str(zero_point.dtype),
            }
        )
        rows.append(row)

    return sorted(rows, key=lambda row: int(row["layer"]))


def inspect_model(
    *,
    model_path: Path,
    expected_layers: int = 12,
    fail_scale_ge: float | None = None,
) -> dict[str, object]:
    onnx_path = _find_single_onnx(model_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    rows = _find_softmax_input_qdq(model.graph)

    found_rows = [row for row in rows if row.get("found_qdq")]
    scales = [
        float(row["scale"])
        for row in found_rows
        if row.get("scale") is not None
    ]
    summary = {
        "model": str(onnx_path),
        "expected_layers": expected_layers,
        "num_softmax_layers": len(rows),
        "num_qdq_pairs": len(found_rows),
        "rows": rows,
        "max_add_output_scale": max(scales) if scales else None,
        "min_add_output_scale": min(scales) if scales else None,
    }

    errors: list[str] = []
    if len(rows) != expected_layers:
        errors.append(f"expected {expected_layers} text attention Softmax nodes, found {len(rows)}")
    if len(found_rows) != expected_layers:
        errors.append(f"expected {expected_layers} Softmax-input QDQ pairs, found {len(found_rows)}")
    if fail_scale_ge is not None and scales and max(scales) >= fail_scale_ge:
        errors.append(
            f"max Add_output_0 scale {max(scales):.6g} >= fail threshold {fail_scale_ge:.6g}"
        )
    summary["errors"] = errors
    summary["pass"] = not errors
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect QDQ scales on text self-attention scores+mask tensors."
    )
    parser.add_argument("--model", type=Path, required=True, help="QDQ ONNX directory or file.")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=12)
    parser.add_argument("--fail-scale-ge", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_model(
        model_path=args.model.expanduser().resolve(),
        expected_layers=args.expected_layers,
        fail_scale_ge=args.fail_scale_ge,
    )
    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.expanduser().resolve().write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

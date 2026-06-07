#!/usr/bin/env python3
"""Summarize QDQ quantization encodings in an ONNX model."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path

    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one .onnx file in {path}, found: {candidates}"
        )
    return candidates[0]


def _parse_int_csv(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            result.add(int(item))
    return result


def _node_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output}


def _find_qdq_pairs(
    graph: onnx.GraphProto,
) -> dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]]:
    producer = _node_by_output(graph)
    pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]] = {}
    for dq_node in graph.node:
        if dq_node.op_type != "DequantizeLinear" or not dq_node.input:
            continue
        q_node = producer.get(dq_node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear":
            continue
        if len(q_node.output) != 1 or len(dq_node.output) != 1:
            continue
        pairs[dq_node.output[0]] = (
            q_node,
            dq_node,
            q_node.input[0],
            dq_node.output[0],
        )
    return pairs


def _encoder_block_start_indices(graph: onnx.GraphProto) -> dict[int, int]:
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


def _block_ranges(graph: onnx.GraphProto) -> dict[int, tuple[int, int]]:
    starts = _encoder_block_start_indices(graph)
    post_start = _post_layernorm_start_index(graph)
    ranges: dict[int, tuple[int, int]] = {}
    for block, start in starts.items():
        next_starts = [index for number, index in starts.items() if number > block]
        end = min(next_starts) if next_starts else post_start or len(graph.node)
        ranges[block] = (start, end)
    return ranges


def _block_for_index(index: int | None, ranges: dict[int, tuple[int, int]]) -> int | None:
    if index is None:
        return None
    for block, (start, end) in ranges.items():
        if start <= index < end:
            return block
    return None


def _quant_bounds(dtype: np.dtype) -> tuple[int, int]:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    raise ValueError(f"Unsupported zero-point dtype for quant bounds: {dtype}")


def _array_stats(values: np.ndarray) -> dict[str, object]:
    flat = values.astype(np.float64).reshape(-1)
    return {
        "shape": list(values.shape),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def _encoding_stats(
    initializers: dict[str, np.ndarray],
    q_node: onnx.NodeProto,
) -> dict[str, object]:
    scale_name = q_node.input[1] if len(q_node.input) > 1 else ""
    zp_name = q_node.input[2] if len(q_node.input) > 2 else ""
    scale = initializers.get(scale_name)
    zero_point = initializers.get(zp_name)

    result: dict[str, object] = {
        "scale_name": scale_name,
        "zero_point_name": zp_name,
        "scale_found": scale is not None,
        "zero_point_found": zero_point is not None,
    }
    if scale is None or zero_point is None:
        return result

    qmin, qmax = _quant_bounds(zero_point.dtype)
    scale64 = scale.astype(np.float64)
    zp64 = zero_point.astype(np.float64)
    real_min = (qmin - zp64) * scale64
    real_max = (qmax - zp64) * scale64
    real_abs_max = np.maximum(np.abs(real_min), np.abs(real_max))

    result.update(
        {
            "scale": _array_stats(scale),
            "zero_point": {
                **_array_stats(zero_point),
                "dtype": str(zero_point.dtype),
            },
            "quant_min": qmin,
            "quant_max": qmax,
            "real_min": _array_stats(real_min),
            "real_max": _array_stats(real_max),
            "real_abs_max": _array_stats(real_abs_max),
        }
    )
    return result


def _row_from_pair(
    *,
    dq_output: str,
    q_node: onnx.NodeProto,
    dq_node: onnx.NodeProto,
    float_tensor: str,
    source_kind: str,
    block: int | None,
    encoding: dict[str, object],
) -> dict[str, object]:
    scale = encoding.get("scale", {})
    zero_point = encoding.get("zero_point", {})
    real_min = encoding.get("real_min", {})
    real_max = encoding.get("real_max", {})
    real_abs_max = encoding.get("real_abs_max", {})
    return {
        "dq_output": dq_output,
        "q_node": q_node.name,
        "dq_node": dq_node.name,
        "float_tensor": float_tensor,
        "source_kind": source_kind,
        "block": "" if block is None else block,
        "scale_name": encoding.get("scale_name", ""),
        "scale_shape": scale.get("shape", ""),
        "scale_min": scale.get("min", ""),
        "scale_mean": scale.get("mean", ""),
        "scale_max": scale.get("max", ""),
        "zero_point_name": encoding.get("zero_point_name", ""),
        "zero_point_dtype": zero_point.get("dtype", ""),
        "zero_point_min": zero_point.get("min", ""),
        "zero_point_max": zero_point.get("max", ""),
        "quant_min": encoding.get("quant_min", ""),
        "quant_max": encoding.get("quant_max", ""),
        "real_min_min": real_min.get("min", ""),
        "real_min_mean": real_min.get("mean", ""),
        "real_max_mean": real_max.get("mean", ""),
        "real_max_max": real_max.get("max", ""),
        "real_abs_max_mean": real_abs_max.get("mean", ""),
        "real_abs_max_max": real_abs_max.get("max", ""),
    }


def _group_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    scale_values = np.array([float(row["scale_mean"]) for row in rows], dtype=np.float64)
    abs_values = np.array(
        [float(row["real_abs_max_mean"]) for row in rows], dtype=np.float64
    )
    return {
        "count": len(rows),
        "scale_mean_min": float(scale_values.min()),
        "scale_mean_mean": float(scale_values.mean()),
        "scale_mean_max": float(scale_values.max()),
        "real_abs_max_mean_min": float(abs_values.min()),
        "real_abs_max_mean_mean": float(abs_values.mean()),
        "real_abs_max_mean_max": float(abs_values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze QDQ scale/zero-point encodings in an ONNX model."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--blocks", default="4,5,6,7,8,9,10,11")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = _find_single_onnx(args.model.expanduser().resolve())
    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    initializer_names = set(initializers)
    producer = _node_by_output(graph)
    node_index_by_name = {node.name: index for index, node in enumerate(graph.node)}
    ranges = _block_ranges(graph)
    requested_blocks = _parse_int_csv(args.blocks)

    rows: list[dict[str, object]] = []
    for dq_output, (q_node, dq_node, float_tensor, _dq_tensor) in sorted(
        _find_qdq_pairs(graph).items()
    ):
        source_kind = "weight" if float_tensor in initializer_names else "activation"
        q_index = node_index_by_name.get(q_node.name)
        producer_node = producer.get(float_tensor)
        producer_index = (
            node_index_by_name.get(producer_node.name) if producer_node else None
        )
        block = _block_for_index(q_index, ranges)
        if block is None:
            block = _block_for_index(producer_index, ranges)

        encoding = _encoding_stats(initializers, q_node)
        rows.append(
            _row_from_pair(
                dq_output=dq_output,
                q_node=q_node,
                dq_node=dq_node,
                float_tensor=float_tensor,
                source_kind=source_kind,
                block=block,
                encoding=encoding,
            )
        )

    activation_rows = [row for row in rows if row["source_kind"] == "activation"]
    requested_rows = [
        row
        for row in activation_rows
        if row["block"] != "" and int(row["block"]) in requested_blocks
    ]
    by_block = {
        str(block): _group_stats(
            [
                row
                for row in activation_rows
                if row["block"] != "" and int(row["block"]) == block
            ]
        )
        for block in sorted(requested_blocks)
    }
    top_abs_ranges = sorted(
        requested_rows,
        key=lambda row: float(row["real_abs_max_mean"] or 0),
        reverse=True,
    )[:20]

    summary = {
        "model": str(onnx_path),
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "num_qdq_pairs": len(rows),
        "num_activation_qdq_pairs": len(activation_rows),
        "requested_blocks": sorted(requested_blocks),
        "requested_activation_qdq_stats": _group_stats(requested_rows),
        "by_block": by_block,
        "top_requested_activation_ranges": top_abs_ranges,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

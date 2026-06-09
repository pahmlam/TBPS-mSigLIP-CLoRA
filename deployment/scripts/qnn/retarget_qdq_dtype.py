#!/usr/bin/env python3
"""Retarget selected QDQ pairs to another integer dtype while preserving range."""

from __future__ import annotations

import argparse
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


def _node_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output}


def _find_qdq_pairs(
    graph: onnx.GraphProto,
) -> dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str]]:
    producer = _node_by_output(graph)
    pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str]] = {}
    for dq_node in graph.node:
        if dq_node.op_type != "DequantizeLinear" or not dq_node.input:
            continue
        q_node = producer.get(dq_node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear":
            continue
        if len(q_node.output) != 1 or len(dq_node.output) != 1:
            continue
        pairs[dq_node.output[0]] = (q_node, dq_node, q_node.input[0])
    return pairs


def _consumers_by_input(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _target_dtype(value: str) -> np.dtype:
    mapping = {
        "qint8": np.dtype(np.int8),
        "quint8": np.dtype(np.uint8),
        "qint16": np.dtype(np.int16),
        "quint16": np.dtype(np.uint16),
    }
    return mapping[value]


def _quant_bounds(dtype: np.dtype) -> tuple[int, int]:
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _replace_initializer(
    graph: onnx.GraphProto,
    initializers: dict[str, np.ndarray],
    name: str,
    values: np.ndarray,
) -> None:
    for initializer in graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(values, name=name))
            initializers[name] = values
            return
    raise KeyError(f"Initializer not found: {name}")


def _retarget_encoding(
    scale: np.ndarray,
    zero_point: np.ndarray,
    target_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    old_qmin, old_qmax = _quant_bounds(zero_point.dtype)
    new_qmin, new_qmax = _quant_bounds(target_dtype)

    scale64 = scale.astype(np.float64)
    zero_point64 = zero_point.astype(np.float64)
    real_min = (old_qmin - zero_point64) * scale64
    real_max = (old_qmax - zero_point64) * scale64
    span = real_max - real_min
    eps = np.finfo(np.float32).eps
    invalid = span <= eps
    if np.any(invalid):
        center = (real_min + real_max) * 0.5
        half = np.maximum(np.abs(center), 1.0)
        real_min = np.where(invalid, center - half, real_min)
        real_max = np.where(invalid, center + half, real_max)
        span = real_max - real_min

    new_scale64 = span / float(new_qmax - new_qmin)
    new_zero_point64 = np.rint(new_qmin - real_min / new_scale64)
    new_zero_point64 = np.clip(new_zero_point64, new_qmin, new_qmax)

    return (
        new_scale64.astype(scale.dtype, copy=False),
        new_zero_point64.astype(target_dtype, copy=False),
        {
            "old_zero_point_dtype": str(zero_point.dtype),
            "new_zero_point_dtype": str(target_dtype),
            "old_scale_mean": float(np.mean(scale64)),
            "new_scale_mean": float(np.mean(new_scale64)),
            "real_min_mean": float(np.mean(real_min)),
            "real_max_mean": float(np.mean(real_max)),
        },
    )


def _save_external_data(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Change selected QDQ pair zero-point dtype and rescale encodings."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--match",
        default=".*",
        help=(
            "Regex matched against q node, dq node, Q input tensor, and DQ output "
            "tensor names."
        ),
    )
    parser.add_argument(
        "--consumer-op",
        default="",
        help="Optional comma-separated consumer op types for the DQ output.",
    )
    parser.add_argument(
        "--source-kind",
        choices=("all", "activation", "weight"),
        default="all",
        help="Retarget activation QDQ, weight QDQ, or both.",
    )
    parser.add_argument(
        "--target-dtype",
        choices=("qint8", "quint8", "qint16", "quint16"),
        required=True,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_onnx = _find_single_onnx(args.model.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_onnx = output_dir / src_onnx.name

    model = onnx.load(src_onnx, load_external_data=True)
    graph = model.graph
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    initializer_names = set(initializers)
    consumers = _consumers_by_input(graph)
    consumer_ops = _csv(args.consumer_op)
    matcher = re.compile(args.match)
    target_dtype = _target_dtype(args.target_dtype)

    changed_rows: list[dict[str, object]] = []
    for dq_output, (q_node, dq_node, float_tensor) in sorted(
        _find_qdq_pairs(graph).items()
    ):
        haystack = "\n".join([q_node.name, dq_node.name, float_tensor, dq_output])
        if not matcher.search(haystack):
            continue
        is_weight = float_tensor in initializer_names
        if args.source_kind == "activation" and is_weight:
            continue
        if args.source_kind == "weight" and not is_weight:
            continue
        if consumer_ops and not any(
            consumer.op_type in consumer_ops for consumer in consumers.get(dq_output, [])
        ):
            continue
        if len(q_node.input) < 3:
            continue

        scale_name = q_node.input[1]
        zero_point_name = q_node.input[2]
        scale = initializers.get(scale_name)
        zero_point = initializers.get(zero_point_name)
        if scale is None or zero_point is None:
            continue

        new_scale, new_zero_point, summary = _retarget_encoding(
            scale, zero_point, target_dtype
        )
        _replace_initializer(graph, initializers, scale_name, new_scale)
        _replace_initializer(graph, initializers, zero_point_name, new_zero_point)
        changed_rows.append(
            {
                "q_node": q_node.name,
                "dq_node": dq_node.name,
                "float_tensor": float_tensor,
                "dq_output": dq_output,
                "scale_name": scale_name,
                "zero_point_name": zero_point_name,
                **summary,
            }
        )

    _save_external_data(model, output_onnx)
    metadata: dict[str, object] = {
        "source_model": str(src_onnx),
        "output_model": str(output_onnx),
        "match": args.match,
        "consumer_op": sorted(consumer_ops),
        "source_kind": args.source_kind,
        "target_dtype": args.target_dtype,
        "num_changed_qdq_pairs": len(changed_rows),
        "changed_pairs": changed_rows,
    }
    if args.check:
        onnx.checker.check_model(str(output_onnx))
        metadata["onnx_check"] = "PASS"

    (output_dir / "retarget_qdq_dtype_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

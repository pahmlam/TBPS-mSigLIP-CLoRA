#!/usr/bin/env python3
"""Tune activation QDQ encodings for selected ViT encoder blocks."""

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


def _target_zero_point_dtype(value: str, current: np.dtype) -> np.dtype:
    if value == "keep":
        return np.dtype(current)
    if value == "int16":
        return np.dtype(np.int16)
    if value == "uint16":
        return np.dtype(np.uint16)
    raise ValueError(f"Unsupported target zero-point dtype: {value}")


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


def _check_model(path: Path) -> None:
    onnx.checker.check_model(str(path))


def _get_default_opset(model: onnx.ModelProto) -> int | None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return None


def _bump_default_opset(model: onnx.ModelProto, target_opset: int | None) -> int | None:
    current_opset = _get_default_opset(model)
    if target_opset is None:
        return current_opset
    if current_opset is None:
        raise ValueError("Model has no default ai.onnx opset import")
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version < target_opset:
                opset.version = target_opset
            return int(opset.version)
    return current_opset


def _smoke_load(path: Path) -> dict[str, object]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options)
    return {
        "providers": session.get_providers(),
        "inputs": [
            {
                "name": value.name,
                "shape": value.shape,
                "type": value.type,
            }
            for value in session.get_inputs()
        ],
        "outputs": [
            {
                "name": value.name,
                "shape": value.shape,
                "type": value.type,
            }
            for value in session.get_outputs()
        ],
    }


def _tune_encoding(
    *,
    scale: np.ndarray,
    zero_point: np.ndarray,
    max_abs: float | None,
    target_dtype: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scale64 = scale.astype(np.float64)
    zero_point64 = zero_point.astype(np.float64)
    old_qmin, old_qmax = _quant_bounds(zero_point.dtype)
    new_zero_point_dtype = _target_zero_point_dtype(target_dtype, zero_point.dtype)
    new_qmin, new_qmax = _quant_bounds(new_zero_point_dtype)

    real_min = (old_qmin - zero_point64) * scale64
    real_max = (old_qmax - zero_point64) * scale64
    old_abs = np.maximum(np.abs(real_min), np.abs(real_max))

    if max_abs is None:
        clipped_min = real_min
        clipped_max = real_max
        clamped = np.zeros_like(old_abs, dtype=bool)
    else:
        clipped_min = np.maximum(real_min, -max_abs)
        clipped_max = np.minimum(real_max, max_abs)
        clamped = old_abs > max_abs
    clipped_min = np.minimum(clipped_min, 0.0)
    clipped_max = np.maximum(clipped_max, 0.0)

    span = clipped_max - clipped_min
    eps = np.finfo(np.float32).eps
    invalid = span <= eps
    if np.any(invalid):
        fallback_abs = max_abs if max_abs is not None else np.maximum(old_abs, 1.0)
        clipped_min = np.where(invalid, -fallback_abs, clipped_min)
        clipped_max = np.where(invalid, fallback_abs, clipped_max)
        span = clipped_max - clipped_min

    new_scale64 = span / float(new_qmax - new_qmin)
    new_zero_point64 = np.rint(new_qmin - clipped_min / new_scale64)
    new_zero_point64 = np.clip(new_zero_point64, new_qmin, new_qmax)

    dtype_changed = np.dtype(zero_point.dtype) != new_zero_point_dtype
    summary = {
        "num_values": int(old_abs.size),
        "num_values_clamped": int(np.count_nonzero(clamped)),
        "old_zero_point_dtype": str(zero_point.dtype),
        "new_zero_point_dtype": str(new_zero_point_dtype),
        "dtype_changed": dtype_changed,
        "old_abs_max": float(np.max(old_abs)),
        "new_abs_max": float(max_abs if np.any(clamped) else np.max(old_abs)),
        "old_scale_mean": float(np.mean(scale64)),
        "new_scale_mean": float(np.mean(new_scale64)),
    }

    return (
        new_scale64.astype(scale.dtype, copy=False),
        new_zero_point64.astype(new_zero_point_dtype, copy=False),
        summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune activation QDQ encodings for selected encoder blocks."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="4,5,6,7,8,9,10,11")
    parser.add_argument(
        "--all-activations",
        action="store_true",
        help="Tune every activation QDQ pair instead of filtering by encoder blocks.",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="Optional symmetric absolute clamp applied before re-encoding.",
    )
    parser.add_argument(
        "--target-dtype",
        choices=("keep", "int16", "uint16"),
        default="keep",
        help="Target dtype for selected activation QuantizeLinear zero-points.",
    )
    parser.add_argument(
        "--bump-opset",
        type=int,
        default=None,
        help=(
            "Bump default ai.onnx opset before saving. Required for standard "
            "ONNX int16/uint16 Q/DQ on older source models."
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_abs is None and args.target_dtype == "keep":
        raise ValueError("Either --max-abs or --target-dtype must change the encodings")
    if args.max_abs is not None and args.max_abs <= 0:
        raise ValueError("--max-abs must be positive")

    src_onnx = _find_single_onnx(args.model.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_onnx = output_dir / src_onnx.name

    model = onnx.load(src_onnx, load_external_data=True)
    original_opset = _get_default_opset(model)
    final_opset = _bump_default_opset(model, args.bump_opset)
    if args.target_dtype in {"int16", "uint16"} and (final_opset or 0) < 21:
        raise ValueError(
            "--target-dtype int16/uint16 requires ai.onnx opset >= 21. "
            "Pass --bump-opset 21 for older source models."
        )
    graph = model.graph
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    initializer_names = set(initializers)
    producer = _node_by_output(graph)
    node_index_by_name = {node.name: index for index, node in enumerate(graph.node)}
    ranges = _block_ranges(graph)
    selected_blocks = _parse_int_csv(args.blocks)
    if not selected_blocks and not args.all_activations:
        raise ValueError("--blocks must not be empty unless --all-activations is set")

    selected_pairs = []
    changed_rows = []
    for dq_output, (q_node, dq_node, float_tensor) in sorted(
        _find_qdq_pairs(graph).items()
    ):
        if float_tensor in initializer_names:
            continue
        q_index = node_index_by_name.get(q_node.name)
        producer_node = producer.get(float_tensor)
        producer_index = (
            node_index_by_name.get(producer_node.name) if producer_node else None
        )
        block = _block_for_index(q_index, ranges)
        if block is None:
            block = _block_for_index(producer_index, ranges)
        if not args.all_activations and block not in selected_blocks:
            continue
        if len(q_node.input) < 3:
            continue

        scale_name = q_node.input[1]
        zero_point_name = q_node.input[2]
        scale = initializers.get(scale_name)
        zero_point = initializers.get(zero_point_name)
        if scale is None or zero_point is None:
            continue

        new_scale, new_zero_point, tune_summary = _tune_encoding(
            scale=scale,
            zero_point=zero_point,
            max_abs=args.max_abs,
            target_dtype=args.target_dtype,
        )
        selected_pairs.append(dq_output)
        should_update = bool(
            tune_summary["num_values_clamped"] or tune_summary["dtype_changed"]
        )
        if should_update:
            _replace_initializer(graph, initializers, scale_name, new_scale)
            _replace_initializer(graph, initializers, zero_point_name, new_zero_point)
            changed_rows.append(
                {
                    "dq_output": dq_output,
                    "q_node": q_node.name,
                    "dq_node": dq_node.name,
                    "float_tensor": float_tensor,
                    "block": block,
                    "scale_name": scale_name,
                    "zero_point_name": zero_point_name,
                    **tune_summary,
                }
            )

    _save_external_data(model, output_onnx)

    metadata: dict[str, object] = {
        "source_model": str(src_onnx),
        "output_model": str(output_onnx),
        "blocks": sorted(selected_blocks),
        "all_activations": args.all_activations,
        "max_abs": args.max_abs,
        "target_dtype": args.target_dtype,
        "original_default_opset": original_opset,
        "final_default_opset": final_opset,
        "num_selected_activation_qdq_pairs": len(selected_pairs),
        "num_changed_activation_qdq_pairs": len(changed_rows),
        "changed_pairs": changed_rows,
    }
    if args.check:
        _check_model(output_onnx)
        metadata["onnx_check"] = "PASS"
    if args.smoke_load:
        metadata["onnxruntime_smoke_load"] = _smoke_load(output_onnx)

    metadata_path = output_dir / "qdq_encoding_tuning_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

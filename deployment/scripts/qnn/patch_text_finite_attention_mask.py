#!/usr/bin/env python3
"""Patch text ONNX attention masks from -FLT_MAX to a finite negative value.

AI Hub QDQ can quantize the text attention tensor after `scores + mask` and
before Softmax. If the exported mask uses `torch.finfo(float32).min`, the QDQ
scale for that tensor becomes enormous and destroys the real attention scores.

This script patches only large negative Constant tensors that sit upstream of
text self-attention Softmax inputs. It is intended for deployment ONNX exports;
it does not modify the PyTorch model or training code.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper


BIG_MASK_THRESHOLD = 1.0e20


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path

    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx file in {path}, found {candidates}")
    return candidates[0]


def _copy_onnx_dir(model_path: Path, output_dir: Path) -> Path:
    src_onnx = _find_single_onnx(model_path)
    src_dir = src_onnx.parent
    output_dir = output_dir.expanduser().resolve()
    if output_dir == src_dir.resolve():
        raise ValueError("Refusing to patch in-place; choose a different --output-dir.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in src_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
    return output_dir / src_onnx.name


def _producer_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output}


def _is_text_attention_softmax(node: onnx.NodeProto) -> bool:
    return (
        node.op_type == "Softmax"
        and "/text_model/encoder/layers." in node.name
        and "/self_attn/Softmax" in node.name
    )


def _upstream_nodes(
    tensor_name: str,
    producer: dict[str, onnx.NodeProto],
    max_depth: int,
) -> list[onnx.NodeProto]:
    found: list[onnx.NodeProto] = []
    seen_tensors: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(tensor_name, 0)])

    while queue:
        tensor, depth = queue.popleft()
        if tensor in seen_tensors or depth > max_depth:
            continue
        seen_tensors.add(tensor)
        node = producer.get(tensor)
        if node is None:
            continue
        found.append(node)
        for input_name in node.input:
            queue.append((input_name, depth + 1))
    return found


def _constant_array(node: onnx.NodeProto) -> tuple[AttributeProto, np.ndarray] | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.type == AttributeProto.TENSOR:
            return attr, numpy_helper.to_array(attr.t)
    return None


def _has_big_negative(values: np.ndarray, threshold: float) -> bool:
    if values.size == 0:
        return False
    finite_values = values.astype(np.float64)
    return bool(np.any(np.isfinite(finite_values) & (finite_values < -threshold)))


def _patch_constant(
    node: onnx.NodeProto,
    attr: AttributeProto,
    values: np.ndarray,
    mask_value: float,
    threshold: float,
) -> dict[str, object]:
    before = values.astype(np.float64)
    replace_mask = np.isfinite(before) & (before < -threshold)
    patched = np.where(
        replace_mask,
        np.asarray(mask_value, dtype=values.dtype),
        values,
    ).astype(values.dtype)

    tensor_name = attr.t.name
    attr.t.CopyFrom(numpy_helper.from_array(patched, name=tensor_name))

    return {
        "node": node.name,
        "outputs": list(node.output),
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "old_min": float(np.min(before)),
        "old_max": float(np.max(before)),
        "new_min": float(np.min(patched.astype(np.float64))),
        "new_max": float(np.max(patched.astype(np.float64))),
        "num_values_changed": int(np.count_nonzero(replace_mask)),
    }


def _attention_softmax_records(
    graph: onnx.GraphProto,
    producer: dict[str, onnx.NodeProto],
    max_depth: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for node in graph.node:
        if not _is_text_attention_softmax(node):
            continue
        softmax_input = node.input[0] if node.input else ""
        upstream = _upstream_nodes(softmax_input, producer, max_depth=max_depth)
        records.append(
            {
                "softmax_node": node.name,
                "softmax_input": softmax_input,
                "upstream_ops": Counter(n.op_type for n in upstream),
                "upstream_constants": [
                    {"node": n.name, "outputs": list(n.output)}
                    for n in upstream
                    if n.op_type == "Constant"
                ],
            }
        )
    return records


def _big_negative_constants_on_attention_paths(
    graph: onnx.GraphProto,
    producer: dict[str, onnx.NodeProto],
    threshold: float,
    max_depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in graph.node:
        if not _is_text_attention_softmax(node) or not node.input:
            continue
        for upstream in _upstream_nodes(node.input[0], producer, max_depth=max_depth):
            constant = _constant_array(upstream)
            if constant is None:
                continue
            _attr, values = constant
            if _has_big_negative(values, threshold):
                rows.append(
                    {
                        "softmax_node": node.name,
                        "constant_node": upstream.name,
                        "outputs": list(upstream.output),
                        "min": float(np.min(values.astype(np.float64))),
                        "max": float(np.max(values.astype(np.float64))),
                    }
                )
    return rows


def _patch_model(
    model: onnx.ModelProto,
    mask_value: float,
    threshold: float,
    max_depth: int,
) -> dict[str, object]:
    graph = model.graph
    producer = _producer_by_output(graph)

    candidate_nodes: dict[str, tuple[onnx.NodeProto, AttributeProto, np.ndarray]] = {}
    for node in graph.node:
        if not _is_text_attention_softmax(node) or not node.input:
            continue
        for upstream in _upstream_nodes(node.input[0], producer, max_depth=max_depth):
            constant = _constant_array(upstream)
            if constant is None:
                continue
            attr, values = constant
            if _has_big_negative(values, threshold):
                key = upstream.name or "|".join(upstream.output)
                candidate_nodes[key] = (upstream, attr, values)

    if not candidate_nodes:
        raise RuntimeError(
            "No large negative text attention mask Constant was found upstream of "
            "text self-attention Softmax inputs."
        )

    changed = [
        _patch_constant(node, attr, values, mask_value, threshold)
        for node, attr, values in candidate_nodes.values()
    ]

    producer_after = _producer_by_output(graph)
    remaining = _big_negative_constants_on_attention_paths(
        graph,
        producer_after,
        threshold=threshold,
        max_depth=max_depth,
    )
    if remaining:
        raise RuntimeError(
            "Large negative constants remain on text attention Softmax paths: "
            + json.dumps(remaining, indent=2)
        )

    softmax_records = _attention_softmax_records(graph, producer_after, max_depth=max_depth)
    return {
        "mask_value": mask_value,
        "big_mask_threshold": threshold,
        "max_trace_depth": max_depth,
        "num_constants_changed": len(changed),
        "constants_changed": changed,
        "num_text_attention_softmax": len(softmax_records),
        "softmax_inputs": [
            {
                "softmax_node": row["softmax_node"],
                "softmax_input": row["softmax_input"],
            }
            for row in softmax_records
        ],
        "op_counts": dict(Counter(node.op_type for node in graph.node)),
    }


def _check_model(path: Path) -> None:
    onnx.checker.check_model(str(path))


def _smoke_load(path: Path) -> dict[str, object]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options)
    return {
        "providers": session.get_providers(),
        "inputs": [
            {"name": value.name, "shape": value.shape, "type": value.type}
            for value in session.get_inputs()
        ],
        "outputs": [
            {"name": value.name, "shape": value.shape, "type": value.type}
            for value in session.get_outputs()
        ],
    }


def patch_onnx_file(
    *,
    model_path: Path,
    output_dir: Path,
    mask_value: float = -32.0,
    threshold: float = BIG_MASK_THRESHOLD,
    max_depth: int = 12,
    check: bool = False,
    smoke_load: bool = False,
) -> dict[str, object]:
    output_onnx = _copy_onnx_dir(model_path, output_dir)
    model = onnx.load(str(output_onnx), load_external_data=False)
    summary = _patch_model(
        model,
        mask_value=mask_value,
        threshold=threshold,
        max_depth=max_depth,
    )
    summary.update(
        {
            "source_model": str(_find_single_onnx(model_path.expanduser().resolve())),
            "output_model": str(output_onnx),
        }
    )

    onnx.save(model, str(output_onnx))
    if check:
        _check_model(output_onnx)
        summary["onnx_check"] = "pass"
    if smoke_load:
        summary["onnxruntime_smoke_load"] = _smoke_load(output_onnx)

    summary_path = output_onnx.parent / "finite_mask_patch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch text ONNX attention mask Constant from -FLT_MAX to a finite value."
    )
    parser.add_argument("--model", type=Path, required=True, help="Text ONNX directory or file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-value", type=float, default=-32.0)
    parser.add_argument("--threshold", type=float, default=BIG_MASK_THRESHOLD)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Maximum producer hops to trace upstream from each text attention Softmax input.",
    )
    parser.add_argument("--check", action="store_true", help="Run onnx.checker on output.")
    parser.add_argument(
        "--smoke-load",
        action="store_true",
        help="Load the patched model with ONNX Runtime CPUExecutionProvider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = patch_onnx_file(
        model_path=args.model,
        output_dir=args.output_dir,
        mask_value=args.mask_value,
        threshold=args.threshold,
        max_depth=args.max_depth,
        check=args.check,
        smoke_load=args.smoke_load,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

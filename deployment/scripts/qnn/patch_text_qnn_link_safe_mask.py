#!/usr/bin/env python3
"""Rewrite the text attention-mask subgraph into a QNN-link-safe form.

The f32-mask ONNX export can still contain a redundant
`Expand -> Cast(FLOAT)` path from HuggingFace's `_expand_mask(...).to(dtype)`.
AI Hub may quantize that cast output as `/text_model/Cast_output_0_updated`,
but QNN HTP v68 rejects it as an internal floating-point tensor during link.

This patch is deliberately narrow:

1. Remove the redundant `Expand -> Cast(FLOAT)` materialization of the mask.
2. Replace `Where(Cast(1-mask), -32, 1-mask)` with `(1-mask) * -32`.

For a binary mask, the math is identical:
  if mask=1: (1-mask)*-32 = 0
  if mask=0: (1-mask)*-32 = -32
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path
    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx in {path}, found: {candidates}")
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


def _consumers_by_input(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _cast_to(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _constant_array(node: onnx.NodeProto) -> np.ndarray | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
    return None


def _replace_node_input(graph: onnx.GraphProto, old: str, new: str) -> int:
    changed = 0
    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old:
                node.input[idx] = new
                changed += 1
    return changed


def _remove_value_info(graph: onnx.GraphProto, names: set[str]) -> None:
    kept = [value for value in graph.value_info if value.name not in names]
    del graph.value_info[:]
    graph.value_info.extend(kept)


def _attention_mask_input_type(graph: onnx.GraphProto) -> str | None:
    for graph_input in graph.input:
        if graph_input.name == "attention_mask":
            return TensorProto.DataType.Name(graph_input.type.tensor_type.elem_type)
    return None


def _rewrite_model(model: onnx.ModelProto, mask_value: float) -> dict[str, object]:
    graph = model.graph
    producer = _producer_by_output(graph)
    consumers = _consumers_by_input(graph)
    removed_outputs: set[str] = set()

    cast_node = producer.get("/text_model/Cast_output_0")
    if cast_node is None or cast_node.op_type != "Cast":
        raise RuntimeError("Expected /text_model/Cast -> /text_model/Cast_output_0 was not found.")
    if _cast_to(cast_node) != TensorProto.FLOAT:
        raise RuntimeError("/text_model/Cast is not a Cast(FLOAT) node.")
    if len(cast_node.input) != 1 or len(cast_node.output) != 1:
        raise RuntimeError("/text_model/Cast must have one input and one output.")

    cast_input = cast_node.input[0]
    cast_output = cast_node.output[0]
    rewired_cast_consumers = _replace_node_input(graph, cast_output, cast_input)
    removed_outputs.add(cast_output)

    producer = _producer_by_output(graph)
    expand_node = producer.get(cast_input)
    rewired_expand_consumers = 0
    expand_input = None
    if expand_node is not None and expand_node.op_type == "Expand" and len(expand_node.input) >= 1:
        # For non-causal text self-attention, [B,1,1,L] broadcasts to [B,H,L,L].
        # Avoid materializing the expanded float mask, which QNN may treat as a
        # floating coefficient tensor during link.
        expand_input = expand_node.input[0]
        rewired_expand_consumers = _replace_node_input(graph, cast_input, expand_input)
        removed_outputs.add(cast_input)

    producer = _producer_by_output(graph)
    where_node = producer.get("/text_model/Where_1_output_0")
    if where_node is None or where_node.op_type != "Where":
        raise RuntimeError("Expected /text_model/Where_1 -> /text_model/Where_1_output_0 was not found.")
    if len(where_node.input) != 3 or len(where_node.output) != 1:
        raise RuntimeError("/text_model/Where_1 must have three inputs and one output.")

    sub_input = None
    inferred_mask_value = mask_value
    for input_name in where_node.input:
        parent = producer.get(input_name)
        if parent is not None and parent.op_type == "Sub":
            sub_input = input_name
        if parent is not None and parent.op_type == "Constant":
            values = _constant_array(parent)
            if values is not None and values.size:
                finite = values.astype(np.float64)
                negatives = finite[np.isfinite(finite) & (finite < 0)]
                if negatives.size:
                    inferred_mask_value = float(negatives.min())

    if sub_input is None:
        raise RuntimeError("Could not identify the (1 - attention_mask) input to /text_model/Where_1.")

    scale_output = "/text_model/qnn_link_safe_mask_scale"
    scale_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[scale_output],
        name="/text_model/qnn_link_safe_mask_scale",
        value=numpy_helper.from_array(np.asarray(inferred_mask_value, dtype=np.float32)),
    )
    mul_node = helper.make_node(
        "Mul",
        inputs=[sub_input, scale_output],
        outputs=list(where_node.output),
        name="/text_model/Mul_qnn_link_safe_mask",
    )

    names_to_remove = {
        cast_node.name,
        "/text_model/Cast_1",
        "/text_model/Cast_2",
        where_node.name,
    }
    if expand_node is not None and expand_node.op_type == "Expand":
        names_to_remove.add(expand_node.name)
    new_nodes = []
    inserted = False
    for node in graph.node:
        if node.name == where_node.name:
            new_nodes.extend([scale_node, mul_node])
            inserted = True
            continue
        if node.name in names_to_remove:
            removed_outputs.update(node.output)
            continue
        new_nodes.append(node)
    if not inserted:
        raise RuntimeError("Failed to insert link-safe mask Mul node.")

    del graph.node[:]
    graph.node.extend(new_nodes)
    _remove_value_info(graph, removed_outputs)

    consumers_after = _consumers_by_input(graph)
    remaining_bad_consumers = [
        node.name for node in consumers_after.get(cast_output, [])
        if node.name not in names_to_remove
    ]
    return {
        "attention_mask_input_type": _attention_mask_input_type(graph),
        "removed_nodes": sorted(names_to_remove),
        "removed_outputs": sorted(removed_outputs),
        "rewired_cast_consumers": rewired_cast_consumers,
        "removed_expand_output": cast_input if expand_input is not None else None,
        "expand_replacement": expand_input,
        "rewired_expand_consumers": rewired_expand_consumers,
        "mask_value": inferred_mask_value,
        "remaining_consumers_of_removed_cast_output": remaining_bad_consumers,
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
    mask_value: float,
    check: bool,
    smoke_load: bool,
) -> dict[str, object]:
    output_onnx = _copy_onnx_dir(model_path, output_dir)
    model = onnx.load(str(output_onnx), load_external_data=False)
    summary = _rewrite_model(model, mask_value=mask_value)
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

    summary_path = output_onnx.parent / "qnn_link_safe_mask_patch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch text f32-mask ONNX into a QNN-link-safe mask subgraph."
    )
    parser.add_argument("--model", type=Path, required=True, help="Text ONNX directory or file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-value", type=float, default=-32.0)
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
        check=args.check,
        smoke_load=args.smoke_load,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

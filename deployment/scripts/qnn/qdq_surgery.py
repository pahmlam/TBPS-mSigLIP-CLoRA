#!/usr/bin/env python3
"""Create local mixed-precision QDQ ONNX candidates.

The script bypasses selected QuantizeLinear -> DequantizeLinear pairs in a QDQ
ONNX graph. It is intended as a local fidelity diagnostic before spending time
on QAI Hub compile/link jobs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import onnx
from onnx import helper


FINAL_HEAD_NODES = {
    "node_linear_74",
    "node_layer_norm_25",
    "node_linear_75",
    "node_gelu_12",
    "node_linear_76",
    "node_add_1219",
    "node_select_2",
}

PRESETS = {
    "output_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "final_head_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": FINAL_HEAD_NODES,
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "all_layernorm_float": {
        "keep_output_float": True,
        "keep_op_float": {"LayerNormalization"},
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "attention_score_float": {
        "keep_output_float": True,
        "keep_op_float": {"Softmax"},
        "keep_node_float": set(),
        "keep_softmax_input_float": True,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "all_qdq_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": True,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "all_weights_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": True,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "all_activations_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": True,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "matmul_gemm_weights_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": True,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": False,
    },
    "encoder_blocks_0_3_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": {0, 1, 2, 3},
        "select_post_layernorm_head_activations": False,
    },
    "encoder_blocks_4_7_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": {4, 5, 6, 7},
        "select_post_layernorm_head_activations": False,
    },
    "encoder_blocks_8_11_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": {8, 9, 10, 11},
        "select_post_layernorm_head_activations": False,
    },
    "post_layernorm_head_float": {
        "keep_output_float": True,
        "keep_op_float": set(),
        "keep_node_float": set(),
        "keep_softmax_input_float": False,
        "select_all_qdq": False,
        "select_weight_qdq": False,
        "select_activation_qdq": False,
        "select_matmul_gemm_weight_qdq": False,
        "select_activation_blocks": set(),
        "select_post_layernorm_head_activations": True,
    },
}


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


def _parse_csv(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.add(item)
    return result


def _parse_int_csv(values: list[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.add(int(item))
    return result


def _copy_model_dir(src_onnx: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in src_onnx.parent.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
    return output_dir / src_onnx.name


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


def _node_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output}


def _consumers_by_input(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _find_qdq_pairs(
    graph: onnx.GraphProto,
) -> dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]]:
    """Return pairs keyed by dequantized tensor name."""
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


def _initializer_names(graph: onnx.GraphProto) -> set[str]:
    return {initializer.name for initializer in graph.initializer}


def _select_weight_pairs(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
) -> set[str]:
    initializer_names = _initializer_names(graph)
    return {
        dq_output
        for dq_output, (_q_node, _dq_node, float_tensor, _dq_tensor) in qdq_pairs.items()
        if float_tensor in initializer_names
    }


def _select_activation_pairs(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
) -> set[str]:
    initializer_names = _initializer_names(graph)
    return {
        dq_output
        for dq_output, (_q_node, _dq_node, float_tensor, _dq_tensor) in qdq_pairs.items()
        if float_tensor not in initializer_names
    }


def _select_matmul_gemm_weight_pairs(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
) -> set[str]:
    initializer_names = _initializer_names(graph)
    consumers = _consumers_by_input(graph)
    selected: set[str] = set()

    for dq_output, (_q_node, _dq_node, float_tensor, _dq_tensor) in qdq_pairs.items():
        if float_tensor not in initializer_names:
            continue
        for consumer in consumers.get(dq_output, []):
            if consumer.op_type not in {"MatMul", "Gemm"}:
                continue
            for input_index, input_name in enumerate(consumer.input):
                if input_name == dq_output and input_index == 1:
                    selected.add(dq_output)

    return selected


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


def _select_activation_pairs_by_node_range(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
    start_index: int,
    end_index: int,
) -> set[str]:
    activation_pairs = _select_activation_pairs(graph, qdq_pairs)
    producer = _node_by_output(graph)
    node_index_by_name = {node.name: index for index, node in enumerate(graph.node)}
    selected: set[str] = set()

    for dq_output in activation_pairs:
        q_node, _dq_node, float_tensor, _dq_tensor = qdq_pairs[dq_output]
        q_index = node_index_by_name.get(q_node.name)
        producer_node = producer.get(float_tensor)
        producer_index = (
            node_index_by_name.get(producer_node.name) if producer_node else None
        )
        q_in_range = q_index is not None and start_index <= q_index < end_index
        producer_in_range = (
            producer_index is not None and start_index <= producer_index < end_index
        )
        if q_in_range or producer_in_range:
            selected.add(dq_output)

    return selected


def _select_encoder_block_activation_pairs(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
    blocks: set[int],
) -> set[str]:
    starts = _encoder_block_start_indices(graph)
    post_start = _post_layernorm_start_index(graph)
    selected: set[str] = set()

    for block in blocks:
        if block not in starts:
            raise ValueError(f"Could not find encoder block {block} in ONNX graph")
        next_starts = [index for number, index in starts.items() if number > block]
        end_index = min(next_starts) if next_starts else post_start or len(graph.node)
        selected.update(
            _select_activation_pairs_by_node_range(
                graph,
                qdq_pairs,
                starts[block],
                end_index,
            )
        )

    return selected


def _select_post_layernorm_head_activation_pairs(
    graph: onnx.GraphProto,
    qdq_pairs: dict[str, tuple[onnx.NodeProto, onnx.NodeProto, str, str]],
) -> set[str]:
    post_start = _post_layernorm_start_index(graph)
    if post_start is None:
        raise ValueError("Could not find vision_model.post_layernorm in ONNX graph")
    return _select_activation_pairs_by_node_range(
        graph,
        qdq_pairs,
        post_start,
        len(graph.node),
    )


def _selected_pairs(
    model: onnx.ModelProto,
    *,
    keep_op_float: set[str],
    keep_node_float: set[str],
    keep_output_float: bool,
    keep_softmax_input_float: bool,
    select_all_qdq: bool,
    select_weight_qdq: bool,
    select_activation_qdq: bool,
    select_matmul_gemm_weight_qdq: bool,
    select_activation_blocks: set[int],
    select_post_layernorm_head_activations: bool,
) -> set[str]:
    graph = model.graph
    qdq_pairs = _find_qdq_pairs(graph)
    qdq_output_by_float_tensor = {
        float_tensor: dq_output
        for dq_output, (_q_node, _dq_node, float_tensor, _dq_tensor) in qdq_pairs.items()
    }
    selected: set[str] = set()

    if select_all_qdq:
        selected.update(qdq_pairs)
    if select_weight_qdq:
        selected.update(_select_weight_pairs(graph, qdq_pairs))
    if select_activation_qdq:
        selected.update(_select_activation_pairs(graph, qdq_pairs))
    if select_matmul_gemm_weight_qdq:
        selected.update(_select_matmul_gemm_weight_pairs(graph, qdq_pairs))
    if select_activation_blocks:
        selected.update(
            _select_encoder_block_activation_pairs(
                graph,
                qdq_pairs,
                select_activation_blocks,
            )
        )
    if select_post_layernorm_head_activations:
        selected.update(_select_post_layernorm_head_activation_pairs(graph, qdq_pairs))

    graph_outputs = {output.name for output in graph.output}
    if keep_output_float:
        selected.update(name for name in graph_outputs if name in qdq_pairs)

    for node in graph.node:
        if node.op_type in keep_op_float or node.name in keep_node_float:
            selected.update(
                qdq_output_by_float_tensor[output]
                for output in node.output
                if output in qdq_output_by_float_tensor
            )
        if keep_softmax_input_float and node.op_type == "Softmax":
            selected.update(input_name for input_name in node.input if input_name in qdq_pairs)

    return selected


def _bypass_pairs(model: onnx.ModelProto, selected: set[str]) -> dict[str, object]:
    graph = model.graph
    qdq_pairs = _find_qdq_pairs(graph)
    consumers = _consumers_by_input(graph)
    graph_outputs = {output.name for output in graph.output}

    selected = {name for name in selected if name in qdq_pairs}
    removed_nodes: set[str] = set()
    rewired_consumers: dict[str, int] = {}
    output_identities: list[onnx.NodeProto] = []

    for dq_output in sorted(selected):
        q_node, dq_node, float_tensor, _ = qdq_pairs[dq_output]
        removed_nodes.update({q_node.name, dq_node.name})

        for consumer in consumers.get(dq_output, []):
            if consumer.name in removed_nodes:
                continue
            count = 0
            for idx, input_name in enumerate(consumer.input):
                if input_name == dq_output:
                    consumer.input[idx] = float_tensor
                    count += 1
            if count:
                rewired_consumers[consumer.name] = (
                    rewired_consumers.get(consumer.name, 0) + count
                )

        if dq_output in graph_outputs:
            identity_name = f"{dq_node.name}_float_output"
            output_identities.append(
                helper.make_node(
                    "Identity",
                    inputs=[float_tensor],
                    outputs=[dq_output],
                    name=identity_name,
                )
            )

    kept_nodes = [node for node in graph.node if node.name not in removed_nodes]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    graph.node.extend(output_identities)

    return {
        "selected_qdq_pairs": len(selected),
        "removed_nodes": len(removed_nodes),
        "rewired_consumers": rewired_consumers,
        "output_identities": [node.name for node in output_identities],
        "selected_outputs": sorted(selected),
    }


def _check_model(path: Path) -> None:
    onnx.checker.check_model(str(path))


def _smoke_load(path: Path) -> dict[str, object]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return {
        "inputs": [
            {"name": value.name, "type": value.type, "shape": value.shape}
            for value in session.get_inputs()
        ],
        "outputs": [
            {"name": value.name, "type": value.type, "shape": value.shape}
            for value in session.get_outputs()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bypass selected QDQ pairs in a QDQ ONNX model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="QDQ ONNX file or directory containing one .onnx file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the edited ONNX + external data will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=["noop", "surgery", "list"],
        default="surgery",
        help="noop saves a load/save baseline; surgery bypasses selected QDQ pairs.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        action="append",
        default=[],
        help="Named candidate preset. Can be repeated.",
    )
    parser.add_argument(
        "--keep-op-float",
        action="append",
        default=[],
        help="Comma-separated ONNX op types whose immediate output QDQ pairs are bypassed.",
    )
    parser.add_argument(
        "--keep-node-float",
        action="append",
        default=[],
        help="Comma-separated node names whose immediate output QDQ pairs are bypassed.",
    )
    parser.add_argument(
        "--keep-output-float",
        action="store_true",
        help="Bypass final graph-output QDQ pairs while preserving output names.",
    )
    parser.add_argument(
        "--keep-softmax-input-float",
        action="store_true",
        help="Bypass QDQ pairs feeding Softmax inputs.",
    )
    parser.add_argument(
        "--all-qdq-float",
        action="store_true",
        help="Bypass every detected QuantizeLinear -> DequantizeLinear pair.",
    )
    parser.add_argument(
        "--keep-weight-float",
        action="store_true",
        help="Bypass QDQ pairs whose float source tensor is an initializer.",
    )
    parser.add_argument(
        "--keep-activation-float",
        action="store_true",
        help="Bypass QDQ pairs whose float source tensor is not an initializer.",
    )
    parser.add_argument(
        "--keep-matmul-gemm-weight-float",
        action="store_true",
        help="Bypass initializer QDQ pairs feeding MatMul/Gemm input B.",
    )
    parser.add_argument(
        "--keep-encoder-block-activation-float",
        action="append",
        default=[],
        help="Comma-separated encoder block indices whose activation QDQ pairs are bypassed.",
    )
    parser.add_argument(
        "--keep-post-layernorm-head-activation-float",
        action="store_true",
        help="Bypass activation QDQ pairs from post-layernorm through the head.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run onnx.checker.check_model on the written model.",
    )
    parser.add_argument(
        "--smoke-load",
        action="store_true",
        help="Load the written model with ONNX Runtime CPUExecutionProvider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_onnx = _find_single_onnx(args.model.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_onnx = output_dir / src_onnx.name

    model = onnx.load(src_onnx, load_external_data=True)
    qdq_pairs = _find_qdq_pairs(model.graph)
    weight_qdq_pairs = _select_weight_pairs(model.graph, qdq_pairs)
    activation_qdq_pairs = _select_activation_pairs(model.graph, qdq_pairs)
    matmul_gemm_weight_qdq_pairs = _select_matmul_gemm_weight_pairs(
        model.graph, qdq_pairs
    )

    keep_op_float = _parse_csv(args.keep_op_float)
    keep_node_float = _parse_csv(args.keep_node_float)
    keep_output_float = args.keep_output_float
    keep_softmax_input_float = args.keep_softmax_input_float
    select_all_qdq = args.all_qdq_float
    select_weight_qdq = args.keep_weight_float
    select_activation_qdq = args.keep_activation_float
    select_matmul_gemm_weight_qdq = args.keep_matmul_gemm_weight_float
    select_activation_blocks = _parse_int_csv(args.keep_encoder_block_activation_float)
    select_post_layernorm_head_activations = (
        args.keep_post_layernorm_head_activation_float
    )
    for preset_name in args.preset:
        preset = PRESETS[preset_name]
        keep_op_float.update(preset["keep_op_float"])
        keep_node_float.update(preset["keep_node_float"])
        keep_output_float = keep_output_float or bool(preset["keep_output_float"])
        keep_softmax_input_float = keep_softmax_input_float or bool(
            preset["keep_softmax_input_float"]
        )
        select_all_qdq = select_all_qdq or bool(preset["select_all_qdq"])
        select_weight_qdq = select_weight_qdq or bool(preset["select_weight_qdq"])
        select_activation_qdq = select_activation_qdq or bool(
            preset["select_activation_qdq"]
        )
        select_matmul_gemm_weight_qdq = select_matmul_gemm_weight_qdq or bool(
            preset["select_matmul_gemm_weight_qdq"]
        )
        select_activation_blocks.update(preset["select_activation_blocks"])
        select_post_layernorm_head_activations = (
            select_post_layernorm_head_activations
            or bool(preset["select_post_layernorm_head_activations"])
        )

    if args.mode == "list":
        summary = {
            "source_model": str(src_onnx),
            "num_nodes": len(model.graph.node),
            "num_qdq_pairs": len(qdq_pairs),
            "num_weight_qdq_pairs": len(weight_qdq_pairs),
            "num_activation_qdq_pairs": len(activation_qdq_pairs),
            "num_matmul_gemm_weight_qdq_pairs": len(
                matmul_gemm_weight_qdq_pairs
            ),
            "graph_outputs": [output.name for output in model.graph.output],
            "op_counts": {},
            "presets": sorted(PRESETS),
        }
        for node in model.graph.node:
            summary["op_counts"][node.op_type] = (
                summary["op_counts"].get(node.op_type, 0) + 1
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.mode == "noop":
        metadata: dict[str, object] = {
            "source_model": str(src_onnx),
            "output_model": str(output_onnx),
            "mode": args.mode,
            "num_qdq_pairs": len(qdq_pairs),
            "selected_qdq_pairs": 0,
        }
    else:
        selected = _selected_pairs(
            model,
            keep_op_float=keep_op_float,
            keep_node_float=keep_node_float,
            keep_output_float=keep_output_float,
            keep_softmax_input_float=keep_softmax_input_float,
            select_all_qdq=select_all_qdq,
            select_weight_qdq=select_weight_qdq,
            select_activation_qdq=select_activation_qdq,
            select_matmul_gemm_weight_qdq=select_matmul_gemm_weight_qdq,
            select_activation_blocks=select_activation_blocks,
            select_post_layernorm_head_activations=select_post_layernorm_head_activations,
        )
        surgery = _bypass_pairs(model, selected)
        metadata = {
            "source_model": str(src_onnx),
            "output_model": str(output_onnx),
            "mode": args.mode,
            "presets": args.preset,
            "keep_op_float": sorted(keep_op_float),
            "keep_node_float": sorted(keep_node_float),
            "keep_output_float": keep_output_float,
            "keep_softmax_input_float": keep_softmax_input_float,
            "select_all_qdq": select_all_qdq,
            "select_weight_qdq": select_weight_qdq,
            "select_activation_qdq": select_activation_qdq,
            "select_matmul_gemm_weight_qdq": select_matmul_gemm_weight_qdq,
            "select_activation_blocks": sorted(select_activation_blocks),
            "select_post_layernorm_head_activations": (
                select_post_layernorm_head_activations
            ),
            "num_qdq_pairs_before": len(qdq_pairs),
            "num_weight_qdq_pairs_before": len(weight_qdq_pairs),
            "num_activation_qdq_pairs_before": len(activation_qdq_pairs),
            "num_matmul_gemm_weight_qdq_pairs_before": len(
                matmul_gemm_weight_qdq_pairs
            ),
            **surgery,
        }

    _save_external_data(model, output_onnx)

    if args.check:
        _check_model(output_onnx)
        metadata["onnx_check"] = "PASS"
    if args.smoke_load:
        metadata["onnxruntime_smoke_load"] = _smoke_load(output_onnx)

    metadata_path = output_dir / "qdq_surgery_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

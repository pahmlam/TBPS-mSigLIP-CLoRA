import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from deployment.scripts.qnn.inspect_text_attention_qdq import inspect_model
from deployment.scripts.qnn.patch_text_finite_attention_mask import patch_onnx_file


def _save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _make_mask_model() -> onnx.ModelProto:
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 2, 2])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 1, 2, 2])
    zero = helper.make_tensor_value_info("zero", TensorProto.FLOAT, [1, 1, 2, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2, 2])

    mask_const = helper.make_node(
        "Constant",
        [],
        ["mask_const"],
        name="/text_model/Constant_14",
        value=numpy_helper.from_array(
            np.asarray(-np.finfo(np.float32).max, dtype=np.float32)
        ),
    )
    where = helper.make_node(
        "Where",
        ["cond", "mask_const", "zero"],
        ["mask"],
        name="/text_model/Where_1",
    )
    add_output = "/text_model/encoder/layers.0/self_attn/Add_output_0"
    add = helper.make_node(
        "Add",
        ["scores", "mask"],
        [add_output],
        name="/text_model/encoder/layers.0/self_attn/Add",
    )
    softmax = helper.make_node(
        "Softmax",
        [add_output],
        ["out"],
        name="/text_model/encoder/layers.0/self_attn/Softmax",
        axis=-1,
    )
    graph = helper.make_graph(
        [mask_const, where, add, softmax],
        "mask_patch_test",
        [scores, cond, zero],
        [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    return model


def _make_qdq_model(scale_value: float) -> onnx.ModelProto:
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 2, 2])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 2, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2, 2])

    add_output = "/text_model/encoder/layers.0/self_attn/Add_output_0"
    add = helper.make_node(
        "Add",
        ["scores", "mask"],
        [add_output],
        name="/text_model/encoder/layers.0/self_attn/Add",
    )
    q = helper.make_node(
        "QuantizeLinear",
        [add_output, "add_scale", "add_zero_point"],
        [add_output + "_q"],
        name="QcQuantizeOp_" + add_output + "_q",
    )
    dq = helper.make_node(
        "DequantizeLinear",
        [add_output + "_q", "add_scale", "add_zero_point"],
        [add_output + "_updated"],
        name="QcQuantizeOp_" + add_output + "_dq",
    )
    softmax = helper.make_node(
        "Softmax",
        [add_output + "_updated"],
        ["out"],
        name="/text_model/encoder/layers.0/self_attn/Softmax",
        axis=-1,
    )
    graph = helper.make_graph(
        [add, q, dq, softmax],
        "qdq_inspect_test",
        [scores, mask],
        [out],
        initializer=[
            numpy_helper.from_array(np.asarray(scale_value, dtype=np.float32), "add_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "add_zero_point"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    return model


class TextFiniteAttentionMaskTest(unittest.TestCase):
    def test_patch_replaces_large_negative_mask_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src" / "model.onnx"
            out_dir = tmp_path / "patched"
            _save_model(_make_mask_model(), src)

            summary = patch_onnx_file(
                model_path=src,
                output_dir=out_dir,
                mask_value=-32.0,
                check=True,
                smoke_load=True,
            )

            self.assertEqual(summary["num_constants_changed"], 1)
            patched = onnx.load(out_dir / "model.onnx", load_external_data=False)
            constants = [node for node in patched.graph.node if node.op_type == "Constant"]
            self.assertEqual(len(constants), 1)
            value_attr = next(attr for attr in constants[0].attribute if attr.name == "value")
            value = numpy_helper.to_array(value_attr.t)
            self.assertEqual(float(value), -32.0)

    def test_inspector_fails_large_softmax_input_qdq_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.onnx"
            _save_model(_make_qdq_model(scale_value=1.0e32), model_path)

            summary = inspect_model(
                model_path=model_path,
                expected_layers=1,
                fail_scale_ge=10.0,
            )

            self.assertFalse(summary["pass"])
            self.assertEqual(summary["num_qdq_pairs"], 1)
            self.assertGreaterEqual(summary["max_add_output_scale"], 10.0)


if __name__ == "__main__":
    unittest.main()

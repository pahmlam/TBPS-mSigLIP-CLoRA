#!/usr/bin/env python3
"""Run ONNX Runtime static QDQ quantization on mSigLIP vision ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


INPUT_SHAPE = (1, 3, 256, 256)


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


def _parse_input_list(input_list: Path, limit: int | None) -> list[Path]:
    root = input_list.parent
    raw_paths: list[Path] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":=" in line:
            _, value = line.split(":=", 1)
        else:
            value = line
        raw_path = Path(value)
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        raw_paths.append(raw_path.resolve())
        if limit is not None and len(raw_paths) >= limit:
            break
    if not raw_paths:
        raise ValueError(f"No raw calibration inputs found in {input_list}")
    return raw_paths


def _load_raw_image(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(INPUT_SHAPE))
    if values.size != expected:
        raise ValueError(
            f"Raw file {path} has {values.size} float32 values, expected {expected}"
        )
    return values.reshape(INPUT_SHAPE)


class RawImageCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, raw_paths: list[Path]) -> None:
        self.input_name = input_name
        self.raw_paths = raw_paths
        self._iterator: Iterator[Path] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = iter(self.raw_paths)
        try:
            raw_path = next(self._iterator)
        except StopIteration:
            return None
        return {self.input_name: _load_raw_image(raw_path)}

    def rewind(self) -> None:
        self._iterator = iter(self.raw_paths)


def _quant_type(value: str) -> QuantType:
    mapping = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
        "qint16": QuantType.QInt16,
        "quint16": QuantType.QUInt16,
    }
    return mapping[value]


def _calibration_method(value: str) -> CalibrationMethod:
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    return mapping[value]


def _prepare_source_model(
    model_path: Path,
    output_dir: Path,
    target_opset: int | None,
) -> Path:
    if target_opset is None:
        return model_path

    model = onnx.load(model_path, load_external_data=True)
    current_opset = None
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            current_opset = opset.version
            if opset.version < target_opset:
                opset.version = target_opset
            break
    if current_opset is None:
        raise ValueError("Model has no default ai.onnx opset import")

    prepared_dir = output_dir / "_source_model"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / model_path.name
    data_path = prepared_path.with_suffix(prepared_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        prepared_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return prepared_path


def _csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize vision ONNX with ONNX Runtime static QDQ quantization."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/deployment/exports/exported_model/vision_onnx"),
        help="FP32 ONNX model directory or file.",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=Path("artifacts/deployment/qnn_inputs/vn3k_train_calib_2000/input_list.txt"),
        help="QNN-style input_list.txt containing float32 raw calibration files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-name", default="image")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--quant-format",
        choices=("qdq", "qoperator"),
        default="qdq",
        help="Quantized ONNX representation to emit.",
    )
    parser.add_argument(
        "--activation-type",
        choices=("qint8", "quint8", "qint16", "quint16"),
        default="quint8",
    )
    parser.add_argument(
        "--weight-type",
        choices=("qint8", "quint8", "qint16", "quint16"),
        default="qint8",
    )
    parser.add_argument(
        "--calibrate-method",
        choices=("minmax", "entropy", "percentile", "distribution"),
        default="minmax",
    )
    parser.add_argument(
        "--op-types",
        default="",
        help="Comma-separated op types to quantize. Empty lets ORT choose defaults.",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        default="",
        help="Comma-separated ONNX node names to exclude from quantization.",
    )
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument(
        "--bump-opset",
        type=int,
        default=None,
        help=(
            "Bump default ai.onnx opset import before quantization. Useful for "
            "standard ONNX 16-bit Q/DQ, which requires opset >= 21."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.999,
        help="Percentile value used only with --calibrate-method percentile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_model_path = _find_single_onnx(args.model.expanduser().resolve())
    input_list = args.input_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"
    model_path = _prepare_source_model(source_model_path, output_dir, args.bump_opset)

    raw_paths = _parse_input_list(input_list, args.num_samples)
    reader = RawImageCalibrationReader(args.input_name, raw_paths)

    extra_options: dict[str, object] = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "DedicatedQDQPair": True,
        "AddQDQPairToWeight": True,
    }
    if args.calibrate_method == "percentile":
        extra_options["CalibPercentile"] = args.percentile

    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=(
            QuantFormat.QOperator
            if args.quant_format == "qoperator"
            else QuantFormat.QDQ
        ),
        op_types_to_quantize=_csv(args.op_types),
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        activation_type=_quant_type(args.activation_type),
        weight_type=_quant_type(args.weight_type),
        nodes_to_exclude=_csv(args.nodes_to_exclude),
        use_external_data_format=True,
        calibrate_method=_calibration_method(args.calibrate_method),
        extra_options=extra_options,
    )

    onnx.checker.check_model(str(output_path))

    metadata = {
        "source_model": str(model_path),
        "original_source_model": str(source_model_path),
        "output_model": str(output_path),
        "input_list": str(input_list),
        "num_samples": len(raw_paths),
        "input_name": args.input_name,
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "quant_format": args.quant_format,
        "calibrate_method": args.calibrate_method,
        "op_types": _csv(args.op_types),
        "nodes_to_exclude": _csv(args.nodes_to_exclude),
        "per_channel": args.per_channel,
        "reduce_range": args.reduce_range,
        "bump_opset": args.bump_opset,
        "extra_options": extra_options,
        "first_raw": str(raw_paths[0]),
        "last_raw": str(raw_paths[-1]),
        "onnx_check": "PASS",
    }
    (output_dir / "ort_quantize_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Submit QAI Hub quantize + compile/link jobs for the mSigLIP vision encoder."""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Any


def _parse_input_specs(value: str) -> dict[str, tuple[tuple[int, ...], str]]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("--input-specs must evaluate to a dict")
    return parsed


def _job_label(job: Any) -> str:
    url = getattr(job, "url", None)
    if callable(url):
        url = url()
    if url:
        return str(url)
    return str(job)


def _dtype(name: str):
    import qai_hub as hub

    try:
        return getattr(hub.QuantizeDtype, name.upper())
    except AttributeError as exc:
        choices = ", ".join(member.name.lower() for member in hub.QuantizeDtype)
        raise ValueError(
            f"Unsupported quantize dtype {name!r}; choices: {choices}"
        ) from exc


def _calibration_data(value: str):
    """Return a local dataset path or a QAI Hub Dataset resolved from an ID."""
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    import qai_hub as hub

    try:
        return hub.get_dataset(value)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve calibration data {value!r} as a local path or "
            "QAI Hub dataset ID. Use list_qaihub_datasets.py to verify the ID."
        ) from exc


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path

    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one .onnx file in {path}, found {candidates}"
        )
    return candidates[0]


def _dim_value(dim) -> int | str | None:
    if dim.dim_value:
        return int(dim.dim_value)
    if dim.dim_param:
        return str(dim.dim_param)
    return None


def _shape(value_info) -> list[int | str | None]:
    return [_dim_value(dim) for dim in value_info.type.tensor_type.shape.dim]


def _set_shape(value_info, shape: tuple[int, ...]) -> None:
    dims = value_info.type.tensor_type.shape.dim
    del dims[:]
    for size in shape:
        dim = dims.add()
        dim.dim_value = int(size)


def _prepare_static_onnx(
    model_path: Path,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    static_model_dir: Path | None,
) -> Path:
    """Copy ONNX directory and rewrite model inputs to static shapes."""
    import onnx

    src_onnx = _find_single_onnx(model_path)
    src_dir = src_onnx.parent
    if static_model_dir is None:
        static_model_dir = src_dir.with_name(f"{src_dir.name}_static")
    static_model_dir = static_model_dir.expanduser().resolve()
    static_model_dir.mkdir(parents=True, exist_ok=True)

    for source in src_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, static_model_dir / source.name)

    dst_onnx = static_model_dir / src_onnx.name
    model = onnx.load(dst_onnx, load_external_data=False)
    changed = False

    specs_by_name = {
        name: tuple(int(dim) for dim in spec[0]) for name, spec in input_specs.items()
    }
    for graph_input in model.graph.input:
        if graph_input.name not in specs_by_name:
            continue
        desired_shape = specs_by_name[graph_input.name]
        current_shape = _shape(graph_input)
        if tuple(current_shape) != desired_shape:
            _set_shape(graph_input, desired_shape)
            changed = True
            print(
                f"Staticized input {graph_input.name}: {current_shape} -> {desired_shape}"
            )

    if not changed:
        print("ONNX inputs already match the requested static input specs.")

    onnx.save(model, dst_onnx)
    return static_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production AI Hub flow: submit_quantize_job followed by "
            "submit_compile_and_link_jobs. Use this instead of deprecated "
            "submit-compile-job --quantize_full_type for HTP INT8."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/deployment/exports/exported_model/vision_onnx"),
        help="ONNX model directory or file.",
    )
    parser.add_argument(
        "--calibration-data",
        required=True,
        help="AI Hub dataset ID, e.g. d7x5gzne9.",
    )
    parser.add_argument(
        "--device",
        default="Dragonwing RB3 Gen 2 Vision Kit",
        help="AI Hub target device name.",
    )
    parser.add_argument(
        "--input-specs",
        default='{"image": ((1, 3, 256, 256), "float32")}',
        help="Python literal input specs for compile/link.",
    )
    parser.add_argument(
        "--static-model-dir",
        type=Path,
        help=(
            "Directory for the static-shape ONNX copy used by quantize. "
            "Default: sibling directory named <model_dir>_static."
        ),
    )
    parser.add_argument(
        "--no-staticize",
        action="store_true",
        help="Submit the model as-is. Only use if the ONNX input shapes are already static.",
    )
    parser.add_argument(
        "--weights-dtype",
        default="int8",
        help="Quantized weight dtype. Default: int8.",
    )
    parser.add_argument(
        "--activations-dtype",
        default="int8",
        help="Quantized activation dtype. Default: int8.",
    )
    parser.add_argument(
        "--quantize-options",
        default="",
        help="Additional cli-like options for submit_quantize_job.",
    )
    parser.add_argument(
        "--compile-options",
        default="--quantize_io",
        help=(
            "Additional cli-like compile options. Default asks the new API to "
            "quantize graph I/O instead of preserving FP I/O."
        ),
    )
    parser.add_argument(
        "--link-options",
        default="",
        help="Additional cli-like link options.",
    )
    parser.add_argument(
        "--name",
        default="mSigLIP-vision-int8-vn3k-calib-500-api",
        help="AI Hub job name prefix.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for quantize, compile, and link jobs to finish.",
    )
    parser.add_argument(
        "--download",
        type=Path,
        help="Optional path to download the linked QNN context binary after --wait.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the static ONNX copy and exit without submitting AI Hub jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import qai_hub as hub

    model = args.model.expanduser().resolve()
    input_specs = _parse_input_specs(args.input_specs)
    quantize_model = model
    if not args.no_staticize:
        quantize_model = _prepare_static_onnx(model, input_specs, args.static_model_dir)
        print(f"Static ONNX model directory: {quantize_model}")

    if args.prepare_only:
        return

    print("Submitting quantize job")
    print(f"  model:            {quantize_model}")
    print(f"  calibration_data: {args.calibration_data}")
    calibration_data = _calibration_data(args.calibration_data)
    quantize_job = hub.submit_quantize_job(
        quantize_model,
        calibration_data=calibration_data,
        weights_dtype=_dtype(args.weights_dtype),
        activations_dtype=_dtype(args.activations_dtype),
        name=f"{args.name}-quantize",
        options=args.quantize_options,
    )
    print(f"Quantize job: {_job_label(quantize_job)}")

    quantize_status = None
    if args.wait:
        print("Waiting for quantize job")
        quantize_status = quantize_job.wait()
        print(f"Quantize status: {quantize_status}")

    quantized_model = quantize_job.get_target_model()
    if quantized_model is None:
        raise RuntimeError(
            "Quantize job did not produce a target model. "
            f"Status: {quantize_status or quantize_job.get_status()}"
        )

    print("Submitting compile/link jobs")
    print(f"  device:          {args.device}")
    print(f"  input_specs:     {input_specs}")
    print(f"  compile_options: {args.compile_options!r}")
    compile_jobs, link_job = hub.submit_compile_and_link_jobs(
        quantized_model,
        device=hub.Device(args.device),
        name=f"{args.name}-compile-link",
        input_specs=input_specs,
        compile_options=args.compile_options,
        link_options=args.link_options,
    )

    for index, compile_job in enumerate(compile_jobs):
        print(f"Compile job {index}: {_job_label(compile_job)}")

    if args.wait:
        for index, compile_job in enumerate(compile_jobs):
            print(f"Waiting for compile job {index}")
            print(f"Compile status {index}: {compile_job.wait()}")

    if link_job is None:
        raise RuntimeError(
            "AI Hub did not create a link job; inspect compile job logs."
        )

    print(f"Link job: {_job_label(link_job)}")
    if args.wait:
        print("Waiting for link job")
        print(f"Link status: {link_job.wait()}")

    if args.download:
        if not args.wait:
            raise RuntimeError("--download requires --wait")
        output = args.download.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        downloaded = link_job.download_target_model(str(output))
        print(f"Downloaded linked model: {downloaded or output}")

    print("\nNext checks:")
    print(
        "  1. Download job logs and verify there is no --preserve_io_datatype image output_0."
    )
    print("  2. Transfer the .bin to RB3 and rerun vn3k_test_10.")
    print("  3. Run compare_qnn_with_pytorch.py before any larger benchmark.")


if __name__ == "__main__":
    main()

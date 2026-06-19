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


def _status_code(status: Any) -> str:
    code = getattr(status, "code", status)
    name = getattr(code, "name", None)
    if name is not None:
        return str(name)
    return str(code)


def _raise_if_not_success(status: Any, label: str) -> None:
    code = _status_code(status).upper()
    if code == "SUCCESS" or code.endswith(".SUCCESS"):
        return
    message = getattr(status, "message", "")
    details = f"\n{message}" if message else ""
    raise RuntimeError(f"{label} failed with status {code}.{details}")


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


def _download_model(model: Any, output: Path, label: str) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = model.download(str(output))
    downloaded_path = Path(downloaded).expanduser().resolve() if downloaded else output
    print(f"Downloaded {label}: {downloaded_path}")
    return downloaded_path


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
    allowed_suffixes = {".onnx", ".data", ".encodings", ".bin"}
    if static_model_dir is None:
        static_model_dir = src_dir.with_name(f"{src_dir.name}_static")
    static_model_dir = static_model_dir.expanduser().resolve()
    static_model_dir.mkdir(parents=True, exist_ok=True)

    # QAI Hub rejects model directories with helper metadata such as JSON/CSV.
    # Keep the static copy to model payload files only.
    for existing in static_model_dir.iterdir():
        if existing.is_file() and existing.suffix not in allowed_suffixes:
            existing.unlink()
    for source in src_dir.iterdir():
        if source.is_file() and source.suffix in allowed_suffixes:
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
        "--modality",
        choices=["vision", "text"],
        default="vision",
        help=(
            "Selects safe defaults for --input-specs and --compile-options. "
            "vision: float image input + --quantize_io. text: integer input_ids, "
            "integer-or-float attention_mask, --truncate_64bit_io, and --quantize_io "
            "for HTP-compatible output I/O. Explicit flags override."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Text only. Token sequence length for the int input-specs.",
    )
    parser.add_argument(
        "--text-attention-mask-dtype",
        choices=["int64", "float32"],
        default="int64",
        help=(
            "Text only. Default attention_mask dtype for --input-specs. "
            "Use float32 for text ONNX exports created with "
            "--attention-mask-dtype float32."
        ),
    )
    parser.add_argument(
        "--input-specs",
        default=None,
        help=(
            "Python literal input specs for compile/link. Default by --modality: "
            "vision -> {\"image\": ((1,3,256,256),\"float32\")}; "
            "text -> {\"input_ids\": ((1,S),\"int64\"), "
            "\"attention_mask\": ((1,S), <text attention mask dtype>)}."
        ),
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
        help=(
            "Additional cli-like options for submit_quantize_job. If the value "
            "starts with '--', pass it with equals, e.g. "
            "--quantize-options=--lite_mp."
        ),
    )
    parser.add_argument(
        "--compile-options",
        default=None,
        help=(
            "Additional cli-like compile options. Default by --modality: vision -> "
            "'--quantize_io' (quantize graph I/O); text -> "
            "'--truncate_64bit_io --quantize_io' (truncate integer token inputs "
            "to QNN-supported 32-bit I/O while quantizing the float embedding output)."
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
        "--quantize-only",
        action="store_true",
        help=(
            "Stop after submit_quantize_job produces a QDQ target model. "
            "Use this for the QDQ-vs-PyTorch fidelity gate before compile/link."
        ),
    )
    parser.add_argument(
        "--download-quantized",
        type=Path,
        help="Optional path to download the quantized QDQ target model after --wait.",
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

    if (args.download or args.download_quantized) and not args.wait:
        raise RuntimeError("--download and --download-quantized require --wait")

    model = args.model.expanduser().resolve()

    # Resolve modality-dependent defaults (explicit flags always win).
    if args.input_specs is None:
        if args.modality == "vision":
            args.input_specs = '{"image": ((1, 3, 256, 256), "float32")}'
        else:
            s = args.seq_len
            mask_dtype = args.text_attention_mask_dtype
            args.input_specs = (
                f'{{"input_ids": ((1, {s}), "int64"), '
                f'"attention_mask": ((1, {s}), "{mask_dtype}")}}'
            )
    if args.compile_options is None:
        # Vision quantizes float graph I/O. Text keeps token IDs as integer inputs
        # but still needs quantized float output I/O for HTP context linking.
        args.compile_options = (
            "--quantize_io"
            if args.modality == "vision"
            else "--truncate_64bit_io --quantize_io"
        )

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
        _raise_if_not_success(quantize_status, "Quantize job")

    quantized_model = quantize_job.get_target_model()
    if quantized_model is None:
        raise RuntimeError(
            "Quantize job did not produce a target model. "
            f"Status: {quantize_status or quantize_job.get_status()}"
        )

    if args.download_quantized:
        _download_model(quantized_model, args.download_quantized, "quantized QDQ model")

    if args.quantize_only:
        print("\nQuantize-only flow complete.")
        print("Next checks:")
        print("  1. Extract/download the QDQ ONNX model if needed.")
        print("  2. Run compare_onnx_with_pytorch.py on vn3k_test_10.")
        print("  3. Only compile/link if the QDQ fidelity gate passes.")
        return

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
            compile_status = compile_job.wait()
            print(f"Compile status {index}: {compile_status}")
            _raise_if_not_success(compile_status, f"Compile job {index}")

    if link_job is None:
        raise RuntimeError(
            "AI Hub did not create a link job; inspect compile job logs."
        )

    print(f"Link job: {_job_label(link_job)}")
    if args.wait:
        print("Waiting for link job")
        link_status = link_job.wait()
        print(f"Link status: {link_status}")
        _raise_if_not_success(link_status, "Link job")

    if args.download:
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

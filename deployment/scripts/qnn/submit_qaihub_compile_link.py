#!/usr/bin/env python3
"""Submit a pre-quantized/QDQ ONNX model to AI Hub compile + link jobs."""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Any


ALLOWED_MODEL_SUFFIXES = {".onnx", ".data", ".encodings", ".bin"}


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


def _stage_model_dir(model: Path, output_dir: Path | None) -> Path:
    """Create an AI Hub-compatible model dir containing only accepted files."""
    if model.is_file():
        return model

    if not model.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model}")

    allowed_files = [
        path for path in sorted(model.iterdir())
        if path.is_file() and path.suffix in ALLOWED_MODEL_SUFFIXES
    ]
    if not any(path.suffix == ".onnx" for path in allowed_files):
        raise ValueError(f"No .onnx file found in model directory: {model}")

    unexpected_files = [
        path.name for path in sorted(model.iterdir())
        if path.is_file() and path.suffix not in ALLOWED_MODEL_SUFFIXES
    ]
    if not unexpected_files and output_dir is None:
        return model

    if output_dir is None:
        output_dir = (
            Path("artifacts/deployment/runtime/qaihub_compile_inputs")
            / model.name
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()

    for source in allowed_files:
        shutil.copy2(source, output_dir / source.name)

    if unexpected_files:
        print("Staging clean AI Hub model directory")
        print(f"  ignored files: {', '.join(unexpected_files)}")
    print(f"  staged_model:  {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile/link an existing ONNX or QDQ ONNX model with AI Hub. "
            "Use this after local QDQ fidelity passes and do not re-quantize."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "artifacts/deployment/runtime/ptq_experiments/node_level/"
            "encoder_blocks_4_11_float"
        ),
        help="ONNX model directory or .onnx file to compile/link.",
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
        "--compile-options",
        default="--quantize_io",
        help=(
            "Additional cli-like compile options. Default quantizes graph I/O "
            "for HTP instead of preserving FP I/O."
        ),
    )
    parser.add_argument(
        "--link-options",
        default="",
        help="Additional cli-like link options.",
    )
    parser.add_argument(
        "--name",
        default="msiglip-vision-encoder-blocks-4-11-float-compile-link",
        help="AI Hub job name.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for compile and link jobs to finish.",
    )
    parser.add_argument(
        "--download",
        type=Path,
        help="Optional path to download the linked QNN context binary after --wait.",
    )
    parser.add_argument(
        "--staged-model-dir",
        type=Path,
        help=(
            "Directory for the cleaned model copy submitted to AI Hub. "
            "Default: artifacts/deployment/runtime/qaihub_compile_inputs/<model_dir>."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved arguments and exit without submitting jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download and not args.wait:
        raise RuntimeError("--download requires --wait")

    model = _stage_model_dir(
        args.model.expanduser().resolve(),
        args.staged_model_dir,
    )
    input_specs = _parse_input_specs(args.input_specs)

    print("Compile/link pre-quantized model")
    print(f"  model:           {model}")
    print(f"  device:          {args.device}")
    print(f"  input_specs:     {input_specs}")
    print(f"  compile_options: {args.compile_options!r}")
    print(f"  link_options:    {args.link_options!r}")
    print(f"  name:            {args.name}")

    if args.dry_run:
        return

    import qai_hub as hub

    compile_jobs, link_job = hub.submit_compile_and_link_jobs(
        model,
        device=hub.Device(args.device),
        name=args.name,
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
        raise RuntimeError("AI Hub did not create a link job; inspect compile logs.")

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
        downloaded_path = Path(downloaded).expanduser().resolve() if downloaded else output
        if not downloaded_path.exists():
            raise RuntimeError(f"Download reported success but file is missing: {downloaded_path}")
        print(f"Downloaded linked model: {downloaded_path}")

    print("\nNext checks:")
    print("  1. Inspect AI Hub compile logs for unsupported mixed QDQ/float patterns.")
    print("  2. Transfer the .bin to RB3 and run vn3k_test_10.")
    print("  3. Run compare_qnn_with_pytorch.py before larger benchmark/retrieval.")


if __name__ == "__main__":
    main()

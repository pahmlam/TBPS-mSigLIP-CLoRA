"""
Convert FP32 ONNX models to FP16 with FP16 I/O for Qualcomm HTP deployment.

The Hexagon DSP/HTP on QCS6490 does not support FP32 tensors — not even at I/O
boundaries. AI Hub's `--quantize_full_type float16` compile option converts
internal weights but keeps I/O as FP32 (via --preserve_io_datatype), which HTP
rejects.

This script does the conversion offline using onnxconverter-common, producing
a model where every tensor (weights + I/O + activations) is FP16. The output
can then be compiled to QNN context binary without any quantization flags.

Usage:
    python deployment/scripts/onnx/to_fp16.py \
        --input artifacts/deployment/exports/msiglip_lora/vision_onnx/ \
        --output artifacts/deployment/exports/msiglip_lora/vision_onnx_fp16/

Each *_onnx/ directory is expected to contain one .onnx file (+ its .onnx.data
companion for external weights). The output directory mirrors this layout.
"""

import argparse
import os
import sys

import onnx
from onnxconverter_common import float16

_deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _deployment_root)
from deploy_utils import TeeLogger

_project_root = os.path.dirname(_deployment_root)
DEFAULT_LOG_DIR = os.path.join(_project_root, "artifacts", "deployment", "logs")


def find_onnx(directory: str) -> str:
    """Return the single .onnx file in a directory."""
    candidates = [f for f in os.listdir(directory) if f.endswith(".onnx")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one .onnx file in {directory}, found: {candidates}")
    return os.path.join(directory, candidates[0])


def convert(input_dir: str, output_dir: str) -> None:
    src_onnx = find_onnx(input_dir)
    src_size_mb = os.path.getsize(src_onnx) / 1024**2
    data_path = src_onnx + ".data"
    if os.path.exists(data_path):
        src_size_mb += os.path.getsize(data_path) / 1024**2

    print(f"Loading FP32 ONNX: {src_onnx} ({src_size_mb:.1f} MB total)")
    model = onnx.load(src_onnx, load_external_data=True)

    print("Converting to FP16 (keep_io_types=False → I/O also FP16)...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=False,       # I/O cast to FP16 (required for HTP)
        disable_shape_infer=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    dst_name = os.path.basename(src_onnx)
    dst_onnx = os.path.join(output_dir, dst_name)

    # Save with external data (re-use the same .data filename convention)
    onnx.save_model(
        model_fp16,
        dst_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dst_name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )

    dst_size_mb = os.path.getsize(dst_onnx) / 1024**2
    dst_data = dst_onnx + ".data"
    if os.path.exists(dst_data):
        dst_size_mb += os.path.getsize(dst_data) / 1024**2

    print(f"Saved: {output_dir}/ ({dst_size_mb:.1f} MB total, ~{src_size_mb / dst_size_mb:.1f}x smaller)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="Directory with FP32 ONNX (e.g. artifacts/deployment/exports/msiglip_lora/vision_onnx/)",
    )
    parser.add_argument("--output", required=True, help="Output directory for FP16 ONNX")
    args = parser.parse_args()

    convert(args.input, args.output)

    print("\nDone! Submit the FP16 model to Qualcomm AI Hub:")
    print(
        f"  qai-hub submit-compile-job --model {args.output} "
        f"--device 'Dragonwing RB3 Gen 2 Vision Kit' "
        f"--compile_options ' --target_runtime qnn_context_binary' "
        f"--input_specs '<input_spec with float16>' --wait"
    )


if __name__ == "__main__":
    logger = TeeLogger(DEFAULT_LOG_DIR, "to_fp16")
    main()
    logger.close()

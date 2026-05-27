#!/usr/bin/env python3
"""Upload QNN raw inputs as a Qualcomm AI Hub calibration dataset."""

from __future__ import annotations

import argparse
import sys
from array import array
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_input_list(input_dir: Path, input_list_name: str) -> list[Path]:
    input_list = input_dir / input_list_name
    if not input_list.exists():
        raise FileNotFoundError(f"Input list not found: {input_list}")

    raw_paths: list[Path] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        if ":=" in line:
            _, value = line.split(":=", 1)
        else:
            parts = line.split()
            value = parts[-1] if parts else ""
        raw_path = Path(value)
        if not raw_path.is_absolute():
            raw_path = input_dir / raw_path
        raw_paths.append(raw_path)

    if not raw_paths:
        raise ValueError(f"No raw input paths found in {input_list}")
    return raw_paths


def _read_raw_numpy(path: Path, image_size: int):
    import numpy as np

    expected_values = 3 * image_size * image_size
    data = path.read_bytes()
    if len(data) != expected_values * 4:
        raise ValueError(
            f"{path} has {len(data)} bytes; expected {expected_values * 4} "
            f"for 1x3x{image_size}x{image_size} float32"
        )

    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return np.asarray(values, dtype=np.float32).reshape(1, 3, image_size, image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload prepared VN3K QNN raw tensors as AI Hub calibration data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input_list.txt and raw calibration files.",
    )
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--input-name",
        default="image",
        help="Dataset key. Must match ONNX/QNN model input tensor name.",
    )
    parser.add_argument(
        "--name",
        default="msiglip-vision-vn3k-calibration",
        help="AI Hub dataset name.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap for quick upload tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import qai_hub as hub

    input_dir = args.input_dir.expanduser().resolve()
    raw_paths = _parse_input_list(input_dir, args.input_list)
    if args.max_samples:
        raw_paths = raw_paths[: args.max_samples]

    arrays = [_read_raw_numpy(path, args.image_size) for path in raw_paths]
    dataset = hub.upload_dataset({args.input_name: arrays}, name=args.name)
    dataset_id = getattr(dataset, "dataset_id", None)
    dataset_name = (
        getattr(dataset, "dataset_name", None)
        or getattr(dataset, "name", None)
        or args.name
    )

    print(f"Uploaded {len(arrays)} samples")
    print(f"Dataset name: {dataset_name}")
    if dataset_id:
        print(f"Dataset ID:   {dataset_id}")
        print("\nUse this with:")
        print(f"  --calibration_data {dataset_id}")
    else:
        print("Dataset ID:   <not exposed by this qai_hub client object>")
        print("\nList recent datasets to recover the ID:")
        print(
            "  venv/bin/python deployment/scripts/qnn/list_qaihub_datasets.py "
            "--limit 10"
        )


if __name__ == "__main__":
    main()

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
    parser.add_argument(
        "--modality",
        choices=["vision", "text", "raw"],
        default="vision",
        help=(
            "vision: float32 image .raw under a single key. text: two int keys "
            "(input_ids, attention_mask) parsed from a dual input_list. raw: generic "
            "multi-key dataset driven by --keys (e.g. Gather microbench input_ids, or "
            "split-text inputs_embeds + attention_mask)."
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        help=(
            "raw modality only. One or more 'name:dtype:d0,d1,...' specs matching the "
            "input_list keys, e.g. --keys input_ids:int32:1,64 or "
            "--keys inputs_embeds:float32:1,64,768 attention_mask:float32:1,64."
        ),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64, help="Text only. Token sequence length.")
    parser.add_argument(
        "--id-dtype",
        choices=["int64", "int32"],
        default="int64",
        help="Text only. Integer dtype to upload (must match the text ONNX input type).",
    )
    parser.add_argument(
        "--mask-dtype",
        choices=["same", "int64", "int32", "float32"],
        default="same",
        help=(
            "Text only. attention_mask dtype to upload. Use float32 with a "
            "float32-mask ONNX export."
        ),
    )
    parser.add_argument(
        "--input-name",
        default="image",
        help="Vision only. Dataset key; must match the ONNX input tensor name.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="AI Hub dataset name. Default by modality.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap for quick upload tests.",
    )
    return parser.parse_args()


def _read_raw_float32(path: Path, length: int):
    import numpy as np

    data = path.read_bytes()
    expected = length * np.dtype(np.float32).itemsize
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes does not match {length} float32 values")
    return np.frombuffer(data, dtype=np.float32).reshape(1, length).copy()


def _build_text_dataset(
    input_dir: Path,
    input_list: str,
    seq_len: int,
    id_dtype: str,
    mask_dtype: str,
    max_samples: int | None,
):
    """Parse a dual input_list and return {"input_ids": [...], "attention_mask": [...]}."""
    import sys as _sys

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in _sys.path:
        _sys.path.insert(0, script_dir)
    import numpy as np
    from compare_text_onnx_with_pytorch import _parse_dual_input_list, _read_raw_ints

    rows = _parse_dual_input_list(input_dir, input_list)
    if max_samples:
        rows = rows[:max_samples]
    id_np_dtype = np.int64 if id_dtype == "int64" else np.int32
    resolved_mask_dtype = id_dtype if mask_dtype == "same" else mask_dtype
    mask_np_dtype = {
        "int64": np.int64,
        "int32": np.int32,
        "float32": np.float32,
    }[resolved_mask_dtype]
    ids = [_read_raw_ints(r["input_ids"], seq_len).astype(id_np_dtype) for r in rows]
    if resolved_mask_dtype == "float32":
        masks = [_read_raw_float32(r["attention_mask"], seq_len).astype(mask_np_dtype) for r in rows]
    else:
        masks = [_read_raw_ints(r["attention_mask"], seq_len).astype(mask_np_dtype) for r in rows]
    return {"input_ids": ids, "attention_mask": masks}, len(rows)


def _build_raw_dataset(
    input_dir: Path,
    input_list: str,
    key_specs: list[str],
    max_samples: int | None,
):
    """Generic multi-key dataset from a dual input_list and 'name:dtype:shape' specs."""
    import sys as _sys

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in _sys.path:
        _sys.path.insert(0, script_dir)
    import numpy as np
    from compare_text_onnx_with_pytorch import _parse_dual_input_list

    np_by_name = {"int64": np.int64, "int32": np.int32, "float32": np.float32}
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for spec in key_specs:
        name, dtype_name, shape_str = spec.split(":")
        shape = tuple(int(s) for s in shape_str.split(","))
        specs[name] = (np.dtype(np_by_name[dtype_name]), shape)

    rows = _parse_dual_input_list(input_dir, input_list)
    if max_samples:
        rows = rows[:max_samples]

    data: dict[str, list] = {name: [] for name in specs}
    for row in rows:
        for name, (dtype, shape) in specs.items():
            if name not in row:
                raise SystemExit(f"Key {name!r} not found in input_list row: {row}")
            count = int(np.prod(shape))
            buf = row[name].read_bytes()
            expected = count * dtype.itemsize
            if len(buf) != expected:
                raise ValueError(f"{row[name]}: {len(buf)} bytes != {count} {dtype} values")
            data[name].append(np.frombuffer(buf, dtype=dtype).reshape(shape).copy())
    return data, len(rows)


def main() -> None:
    args = parse_args()

    import qai_hub as hub

    input_dir = args.input_dir.expanduser().resolve()

    if args.modality == "raw":
        if not args.keys:
            raise SystemExit("--modality raw requires --keys name:dtype:shape ...")
        data, count = _build_raw_dataset(input_dir, args.input_list, args.keys, args.max_samples)
        name = args.name or "msiglip-raw-calibration"
    elif args.modality == "text":
        data, count = _build_text_dataset(
            input_dir,
            args.input_list,
            args.seq_len,
            args.id_dtype,
            args.mask_dtype,
            args.max_samples,
        )
        name = args.name or "msiglip-text-vn3k-calibration"
    else:
        raw_paths = _parse_input_list(input_dir, args.input_list)
        if args.max_samples:
            raw_paths = raw_paths[: args.max_samples]
        data = {args.input_name: [_read_raw_numpy(path, args.image_size) for path in raw_paths]}
        count = len(data[args.input_name])
        name = args.name or "msiglip-vision-vn3k-calibration"

    dataset = hub.upload_dataset(data, name=name)
    dataset_id = getattr(dataset, "dataset_id", None)
    dataset_name = (
        getattr(dataset, "dataset_name", None)
        or getattr(dataset, "name", None)
        or name
    )

    print(f"Uploaded {count} samples ({args.modality}, keys={list(data.keys())})")
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

#!/usr/bin/env python3
"""Prepare real VN3K image inputs for QNN vision encoder tests.

The generated raw tensors match the mSigLIP vision preprocessing used for
evaluation:

    RGB -> resize 256x256 -> ToTensor [0, 1] -> Normalize(mean=0.5, std=0.5)

Output layout is NCHW float32 for qnn-net-run's default input parser. The QNN
graph itself may have native uint8/fixed-point I/O; qnn-net-run converts from
float input unless --use_native_input_files is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import stat
import sys
from array import array
from pathlib import Path
from typing import Iterable


def _resolve_dataset_root(path: Path) -> Path:
    """Accept either the VN3K directory or its parent project/data root."""
    path = path.expanduser().resolve()
    if (path / "data_captions_vn3k.json").exists() and (path / "imgs").is_dir():
        return path
    nested = path / "VN3K"
    if (nested / "data_captions_vn3k.json").exists() and (nested / "imgs").is_dir():
        return nested
    raise SystemExit(
        f"Cannot find VN3K annotation/images under {path}. Expected "
        "data_captions_vn3k.json and imgs/."
    )


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "image"


def _load_records(dataset_root: Path, split: str) -> list[dict]:
    ann_path = dataset_root / "data_captions_vn3k.json"
    annos = json.loads(ann_path.read_text(encoding="utf-8"))
    records = []
    for anno in annos:
        anno_split = anno.get("split", "")
        if split != "all" and anno_split != split:
            continue
        image_path = dataset_root / "imgs" / anno["file_path"]
        if not image_path.exists():
            continue
        records.append(
            {
                "pid": int(anno["id"]) - 1,
                "split": anno_split,
                "file_path": anno["file_path"],
                "image_path": image_path,
            }
        )
    if not records:
        raise SystemExit(f"No existing VN3K images found for split={split!r}.")
    return records


def _select_records(
    records: list[dict],
    num_samples: int,
    selection: str,
    seed: int,
    start_index: int,
) -> list[dict]:
    if selection == "random":
        rng = random.Random(seed)
        if num_samples > len(records):
            num_samples = len(records)
        return rng.sample(records, num_samples)

    end = min(start_index + num_samples, len(records))
    return records[start_index:end]


def _resample_filter():
    from PIL import Image

    try:
        return Image.Resampling.BICUBIC
    except AttributeError:  # Pillow < 9
        return Image.BICUBIC


def _preprocess_to_nchw_float32(image_path: Path, size: int) -> tuple[array, tuple[int, int]]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise SystemExit(
            "Pillow is required to decode VN3K images. Install it with: "
            "python3 -m pip install Pillow"
        ) from exc

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        original_size = img.size
        img = img.resize((size, size), _resample_filter())
        pixel_bytes = img.tobytes()

    # ToTensor produces C,H,W and scales to [0, 1]. Normalize(0.5, 0.5) maps
    # to [-1, 1]. Keep channel-major order for ONNX/QNN input [1,3,H,W].
    values = array("f")
    for channel in range(3):
        values.extend(
            (pixel_bytes[index] / 255.0 - 0.5) / 0.5
            for index in range(channel, len(pixel_bytes), 3)
        )

    if sys.byteorder != "little":
        values.byteswap()
    return values, original_size


def _write_input_list(
    records: Iterable[dict],
    output_dir: Path,
    path_mode: str,
    input_name: str,
) -> Path:
    input_list_path = output_dir / "input_list.txt"
    lines = []
    for record in records:
        raw_path = Path(record["raw_path"])
        if path_mode == "absolute":
            listed_path = raw_path.resolve()
        else:
            listed_path = raw_path.relative_to(output_dir)
        lines.append(f"{input_name}:={listed_path}\n")
    input_list_path.write_text("".join(lines), encoding="utf-8")
    return input_list_path


def _write_manifest(records: list[dict], output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "sample_index",
        "pid",
        "split",
        "dataset_file_path",
        "image_path",
        "raw_path",
        "original_width",
        "original_height",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    return manifest_path


def _write_run_script(output_dir: Path, input_list_path: Path) -> Path:
    script_path = output_dir / "run_qnn_vision.sh"
    script = f"""#!/usr/bin/env bash
set -euo pipefail

QAIRT="${{QAIRT:-/opt/qcom/qairt/2.45.40.260406}}"
QNN_BIN="${{QNN_BIN:-$QAIRT/bin/aarch64-ubuntu-gcc9.4}}"
QNN_LIB="${{QNN_LIB:-$QAIRT/lib/aarch64-ubuntu-gcc9.4}}"
export LD_LIBRARY_PATH="$QNN_LIB:${{LD_LIBRARY_PATH:-}}"

# Update this if your skel files live in another 2.45 directory.
if [ -z "${{ADSP_LIBRARY_PATH:-}}" ]; then
  for candidate in "$QAIRT"/lib/hexagon-v68/* "$QAIRT"/lib/hexagon-v68; do
    if ls "$candidate"/*Skel*.so >/dev/null 2>&1; then
      export ADSP_LIBRARY_PATH="$candidate"
      break
    fi
  done
fi

cd "$(dirname "$0")"

"$QNN_BIN/qnn-net-run" \\
  --backend "$QNN_LIB/libQnnHtp.so" \\
  --retrieve_context "${{VISION_BIN:-../../qnn_inputs/vision_encoder.bin}}" \\
  --config_file "${{HTP_CONFIG:-../../../../deployment/config/qnn/htp_config_245.json}}" \\
  --input_list "{input_list_path.name}" \\
  --output_dir "${{OUTPUT_DIR:-../../qnn_runs/vision_results}}" \\
  --profiling_level basic \\
  --perf_profile high_performance \\
  --log_level info
"""
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return script_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VN3K image raw inputs for qnn-net-run vision tests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("VN3K"),
        help="Path to VN3K directory, or its parent. Default: VN3K",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/deployment/qnn_inputs/vn3k_vision"),
        help="Directory for raw files, manifest, and input_list.txt.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "validate", "all"],
        default="test",
        help="VN3K split to sample from.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Use the first N records or a deterministic random sample.",
    )
    parser.add_argument("--seed", type=int, default=2400)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--input-name", default="image")
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Paths written in input_list.txt. Use relative if copying the output dir to RB3.",
    )
    parser.add_argument(
        "--no-run-script",
        action="store_true",
        help="Do not write run_qnn_vision.sh helper script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive.")

    dataset_root = _resolve_dataset_root(args.dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(dataset_root, args.split)
    selected = _select_records(
        records,
        num_samples=args.num_samples,
        selection=args.selection,
        seed=args.seed,
        start_index=args.start_index,
    )

    prepared: list[dict] = []
    for sample_index, record in enumerate(selected):
        values, original_size = _preprocess_to_nchw_float32(
            Path(record["image_path"]), args.image_size
        )
        stem = _sanitize_name(Path(record["file_path"]).with_suffix("").as_posix())
        raw_path = raw_dir / f"{sample_index:05d}_pid{record['pid']}_{stem}.raw"
        with raw_path.open("wb") as f:
            values.tofile(f)
        prepared.append(
            {
                "sample_index": sample_index,
                "pid": record["pid"],
                "split": record["split"],
                "dataset_file_path": record["file_path"],
                "image_path": str(record["image_path"]),
                "raw_path": str(raw_path),
                "original_width": original_size[0],
                "original_height": original_size[1],
            }
        )

    input_list_path = _write_input_list(
        prepared,
        output_dir=output_dir,
        path_mode=args.path_mode,
        input_name=args.input_name,
    )
    manifest_path = _write_manifest(prepared, output_dir)
    run_script_path = None
    if not args.no_run_script:
        run_script_path = _write_run_script(output_dir, input_list_path)

    print(f"Prepared {len(prepared)} VN3K image inputs")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Input list:   {input_list_path}")
    print(f"Manifest:     {manifest_path}")
    if run_script_path:
        print(f"Run script:   {run_script_path}")
    print("\nOn RB3, copy this output dir next to vision_encoder.bin and run:")
    print(f"  cd {output_dir.name}")
    print("  VISION_BIN=../../qnn_inputs/vision_encoder.bin HTP_CONFIG=../../../../deployment/config/qnn/htp_config_245.json ./run_qnn_vision.sh")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize QNN raw image input tensors.

Use this before uploading calibration datasets. It verifies that every raw file
has the expected float32 NCHW shape and reports distribution statistics for one
or more prepared input directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def _stats(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "p01": float(np.percentile(finite_values, 1)),
        "p05": float(np.percentile(finite_values, 5)),
        "p50": float(np.percentile(finite_values, 50)),
        "p95": float(np.percentile(finite_values, 95)),
        "p99": float(np.percentile(finite_values, 99)),
    }


def _summarize_dir(input_dir: Path, input_list_name: str, image_size: int) -> dict:
    input_dir = input_dir.expanduser().resolve()
    raw_paths = _parse_input_list(input_dir, input_list_name)
    expected_values = 3 * image_size * image_size
    expected_bytes = expected_values * 4

    arrays: list[np.ndarray] = []
    file_rows: list[dict] = []
    bad_size_count = 0
    any_nan = False
    any_inf = False

    for index, raw_path in enumerate(raw_paths):
        size_bytes = raw_path.stat().st_size
        valid_size = size_bytes == expected_bytes
        if not valid_size:
            bad_size_count += 1
            values = np.array([], dtype=np.float32)
        else:
            values = np.fromfile(raw_path, dtype="<f4")
            if values.size != expected_values:
                valid_size = False
                bad_size_count += 1
        has_nan = bool(np.isnan(values).any()) if values.size else False
        has_inf = bool(np.isinf(values).any()) if values.size else False
        any_nan = any_nan or has_nan
        any_inf = any_inf or has_inf

        if valid_size:
            arrays.append(values.reshape(3, image_size, image_size))

        file_rows.append(
            {
                "sample_index": index,
                "path": str(raw_path),
                "bytes": size_bytes,
                "valid_size": valid_size,
                "has_nan": has_nan,
                "has_inf": has_inf,
                **_stats(values),
            }
        )

    if arrays:
        stack = np.stack(arrays, axis=0)
        flat = stack.reshape(-1)
        channels = stack.reshape(stack.shape[0], 3, -1)
        channel_stats = {
            f"channel_{channel}": _stats(channels[:, channel, :].reshape(-1))
            for channel in range(3)
        }
        outside_unit = np.mean((flat < -1.000001) | (flat > 1.000001))
    else:
        flat = np.array([], dtype=np.float32)
        channel_stats = {}
        outside_unit = 0.0

    return {
        "input_dir": str(input_dir),
        "input_list": input_list_name,
        "num_listed": len(raw_paths),
        "num_valid": len(arrays),
        "image_size": image_size,
        "expected_shape": [1, 3, image_size, image_size],
        "expected_bytes": expected_bytes,
        "bytes_unique": sorted({row["bytes"] for row in file_rows}),
        "bad_size_count": bad_size_count,
        "any_nan": any_nan,
        "any_inf": any_inf,
        "outside_unit_range_fraction": float(outside_unit),
        "global": _stats(flat),
        "channels": channel_stats,
        "first_file": file_rows[0] if file_rows else None,
        "file_rows": file_rows,
    }


def _write_file_csv(summary: dict, path: Path) -> None:
    rows = summary["file_rows"]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compact(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "file_rows"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prepared float32 raw input tensors."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Prepared input directory containing input_list.txt. Repeatable.",
    )
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--json", type=Path, help="Optional summary JSON path.")
    parser.add_argument(
        "--file-csv-dir",
        type=Path,
        help="Optional directory for per-file CSV summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        _summarize_dir(path, args.input_list, args.image_size)
        for path in args.input_dir
    ]

    result = {"sets": [_compact(summary) for summary in summaries]}
    if len(summaries) >= 2:
        base = summaries[0]["global"]
        comparisons = []
        for summary in summaries[1:]:
            current = summary["global"]
            comparisons.append(
                {
                    "base_input_dir": summaries[0]["input_dir"],
                    "input_dir": summary["input_dir"],
                    "mean_delta": current["mean"] - base["mean"],
                    "std_delta": current["std"] - base["std"],
                    "min_delta": current["min"] - base["min"],
                    "max_delta": current["max"] - base["max"],
                }
            )
        result["comparisons_to_first"] = comparisons

    if args.file_csv_dir:
        csv_dir = args.file_csv_dir.expanduser().resolve()
        csv_dir.mkdir(parents=True, exist_ok=True)
        for summary in summaries:
            name = Path(summary["input_dir"]).name
            _write_file_csv(summary, csv_dir / f"{name}_raw_stats.csv")

    if args.json:
        output = args.json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

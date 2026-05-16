#!/usr/bin/env python3
"""Summarize QNN vision encoder outputs.

This script intentionally uses only the Python standard library so it can run
on minimal host/RB3 environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path


def _result_index(path: Path) -> int:
    try:
        return int(path.parent.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _read_float32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not divisible by 4 bytes: {len(data)}")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def _stats(values: list[float]) -> dict:
    norm = math.sqrt(sum(value * value for value in values))
    return {
        "dim": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values),
        "norm": norm,
        "has_nan": any(math.isnan(value) for value in values),
        "has_inf": any(math.isinf(value) for value in values),
    }


def _load_manifest(path: Path | None) -> list[dict]:
    if not path:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_stats_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "result_index",
        "pid",
        "dataset_file_path",
        "output_path",
        "bytes",
        "dim",
        "min",
        "max",
        "mean",
        "std",
        "norm",
        "has_nan",
        "has_inf",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_embeddings_csv(rows: list[dict], embeddings: list[list[float]], path: Path) -> None:
    dim = len(embeddings[0]) if embeddings else 0
    fieldnames = ["result_index", "pid", "dataset_file_path"] + [
        f"e{i:04d}" for i in range(dim)
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, emb in zip(rows, embeddings):
            out = {
                "result_index": row["result_index"],
                "pid": row.get("pid", ""),
                "dataset_file_path": row.get("dataset_file_path", ""),
            }
            out.update({f"e{i:04d}": value for i, value in enumerate(emb)})
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize qnn-net-run output tensors.")
    parser.add_argument("output_dir", type=Path, help="Directory containing Result_*/output_0.raw")
    parser.add_argument("--manifest", type=Path, help="Optional manifest.csv from prepare_vn3k_vision_inputs.py")
    parser.add_argument("--output-name", default="output_0.raw")
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument(
        "--stats-csv",
        type=Path,
        help="Optional path to write per-output statistics CSV.",
    )
    parser.add_argument(
        "--embeddings-csv",
        type=Path,
        help="Optional path to write L2-normalized embeddings CSV.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(output_dir.glob(f"Result_*/{args.output_name}"), key=_result_index)
    if not files:
        raise SystemExit(f"No Result_*/{args.output_name} files found under {output_dir}")

    manifest_rows = _load_manifest(args.manifest)
    rows: list[dict] = []
    embeddings: list[list[float]] = []
    first_bytes = files[0].read_bytes()
    all_identical = True

    for file_path in files:
        result_index = _result_index(file_path)
        values = _read_float32(file_path)
        if len(values) != args.expected_dim:
            raise SystemExit(
                f"{file_path} has dim={len(values)}, expected {args.expected_dim}"
            )
        stats = _stats(values)
        if file_path.read_bytes() != first_bytes:
            all_identical = False

        norm = stats["norm"]
        normalized = [value / norm for value in values] if norm > 0 else values
        embeddings.append(normalized)

        manifest = manifest_rows[result_index] if result_index < len(manifest_rows) else {}
        rows.append(
            {
                "result_index": result_index,
                "pid": manifest.get("pid", ""),
                "dataset_file_path": manifest.get("dataset_file_path", ""),
                "output_path": str(file_path),
                "bytes": file_path.stat().st_size,
                **stats,
            }
        )

    norms = [row["norm"] for row in rows]
    summary = {
        "output_dir": str(output_dir),
        "num_outputs": len(rows),
        "bytes_per_output_unique": sorted({row["bytes"] for row in rows}),
        "expected_dim": args.expected_dim,
        "all_outputs_byte_identical": all_identical,
        "norm_min": min(norms),
        "norm_max": max(norms),
        "norm_mean": sum(norms) / len(norms),
        "any_nan": any(row["has_nan"] for row in rows),
        "any_inf": any(row["has_inf"] for row in rows),
        "first_output": rows[0],
    }

    if args.stats_csv:
        _write_stats_csv(rows, args.stats_csv)
    if args.embeddings_csv:
        _write_embeddings_csv(rows, embeddings, args.embeddings_csv)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare QNN text encoder outputs against a local PyTorch baseline.

Use this after running the text QNN context binary on RB3. The script reads the
same `input_ids` and `attention_mask` raw files passed to qnn-net-run, maps each
`Result_*/output_0.raw` back to the manifest row, and compares board embeddings
with `TBPS.encode_text`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
QNN_SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, QNN_SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_qnn_with_pytorch import (  # noqa: E402
    _load_manifest,
    _load_pytorch_model,
    _read_float32,
    _result_index,
    _vector_stats,
    _write_json,
)
from compare_text_onnx_with_pytorch import _parse_dual_input_list  # noqa: E402

_NP_DTYPES = {
    "int64": np.int64,
    "int32": np.int32,
    "float32": np.float32,
}


def _read_raw(path: Path, seq_len: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(_NP_DTYPES[dtype_name])
    data = path.read_bytes()
    expected = seq_len * dtype.itemsize
    if len(data) != expected:
        raise ValueError(
            f"{path}: {len(data)} bytes does not match {seq_len} {dtype} values "
            f"({expected} bytes)"
        )
    return np.frombuffer(data, dtype=dtype).reshape(1, seq_len).copy()


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "result_index",
        "pid",
        "caption",
        "input_ids_path",
        "attention_mask_path",
        "qnn_output_path",
        "dim",
        "qnn_norm",
        "torch_norm",
        "cosine_raw",
        "cosine_l2",
        "l2_raw",
        "l2_l2",
        "mean_abs_raw",
        "max_abs_raw",
        "qnn_has_nan",
        "qnn_has_inf",
        "torch_has_nan",
        "torch_has_inf",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare QNN text Result_*/output_0.raw with PyTorch encode_text outputs."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model",
        help="Directory containing config.yaml and model_fp32.pt/model_fp16.pt.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32",
        help="Prepared text input directory containing input_list.txt and raw files.",
    )
    parser.add_argument(
        "--qnn-output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask",
        help="QNN output directory containing Result_*/output_0.raw.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional manifest.csv path.")
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--output-name", default="output_0.raw")
    parser.add_argument("--id-dtype", choices=["int64", "int32"], default="int32")
    parser.add_argument(
        "--mask-dtype",
        choices=["int64", "int32", "float32"],
        default="float32",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/"
        / "qnn_vs_pytorch.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/"
        / "qnn_vs_pytorch_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    qnn_output_dir = args.qnn_output_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else input_dir / "manifest.csv"
    )

    input_rows = _parse_dual_input_list(input_dir, args.input_list)
    qnn_paths = sorted(
        qnn_output_dir.glob(f"Result_*/{args.output_name}"), key=_result_index
    )
    if len(input_rows) != len(qnn_paths):
        raise SystemExit(
            f"Input/output count mismatch: {len(input_rows)} input rows vs "
            f"{len(qnn_paths)} QNN outputs"
        )

    manifest_rows = _load_manifest(manifest_path)
    model, _ = _load_pytorch_model(model_dir, args.precision, args.device)

    rows: list[dict] = []
    with torch.no_grad():
        for idx, (entry, qnn_path) in enumerate(zip(input_rows, qnn_paths)):
            ids = _read_raw(entry["input_ids"], args.seq_len, args.id_dtype)
            attn = _read_raw(entry["attention_mask"], args.seq_len, args.mask_dtype)

            caption_input = {
                "input_ids": torch.from_numpy(ids.astype(np.int64)).to(args.device),
                "attention_mask": torch.from_numpy(attn).to(args.device),
            }
            torch_output = model.encode_text(caption_input).detach().float().cpu().reshape(-1)
            qnn_output = torch.tensor(_read_float32(qnn_path), dtype=torch.float32)

            if qnn_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"{qnn_path} has dim={qnn_output.numel()}, expected {args.expected_dim}"
                )
            if torch_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"PyTorch output for row {idx} has dim={torch_output.numel()}, "
                    f"expected {args.expected_dim}"
                )

            diff = qnn_output - torch_output
            qnn_l2 = F.normalize(qnn_output, dim=0)
            torch_l2 = F.normalize(torch_output, dim=0)
            l2_diff = qnn_l2 - torch_l2
            manifest = manifest_rows[idx] if idx < len(manifest_rows) else {}
            qnn_stats = _vector_stats(qnn_output.tolist())
            torch_stats = _vector_stats(torch_output.tolist())

            rows.append(
                {
                    "result_index": idx,
                    "pid": manifest.get("pid", ""),
                    "caption": (manifest.get("caption", "") or "")[:80],
                    "input_ids_path": str(entry["input_ids"]),
                    "attention_mask_path": str(entry["attention_mask"]),
                    "qnn_output_path": str(qnn_path),
                    "dim": args.expected_dim,
                    "qnn_norm": qnn_stats["norm"],
                    "torch_norm": torch_stats["norm"],
                    "cosine_raw": F.cosine_similarity(
                        qnn_output.unsqueeze(0), torch_output.unsqueeze(0)
                    ).item(),
                    "cosine_l2": torch.dot(qnn_l2, torch_l2).item(),
                    "l2_raw": torch.linalg.vector_norm(diff).item(),
                    "l2_l2": torch.linalg.vector_norm(l2_diff).item(),
                    "mean_abs_raw": torch.mean(torch.abs(diff)).item(),
                    "max_abs_raw": torch.max(torch.abs(diff)).item(),
                    "qnn_has_nan": qnn_stats["has_nan"],
                    "qnn_has_inf": qnn_stats["has_inf"],
                    "torch_has_nan": torch_stats["has_nan"],
                    "torch_has_inf": torch_stats["has_inf"],
                }
            )

    cosines = [row["cosine_l2"] for row in rows]
    l2_l2 = [row["l2_l2"] for row in rows]
    mean_abs = [row["mean_abs_raw"] for row in rows]
    max_abs = [row["max_abs_raw"] for row in rows]
    summary = {
        "model_dir": str(model_dir),
        "input_dir": str(input_dir),
        "qnn_output_dir": str(qnn_output_dir),
        "precision": args.precision,
        "device": args.device,
        "num_samples": len(rows),
        "expected_dim": args.expected_dim,
        "input_dtypes": {
            "input_ids": args.id_dtype,
            "attention_mask": args.mask_dtype,
        },
        "cosine_l2_mean": sum(cosines) / len(cosines),
        "cosine_l2_min": min(cosines),
        "cosine_l2_max": max(cosines),
        "l2_l2_mean": sum(l2_l2) / len(l2_l2),
        "l2_l2_max": max(l2_l2),
        "mean_abs_raw_mean": sum(mean_abs) / len(mean_abs),
        "max_abs_raw_max": max(max_abs),
        "any_qnn_nan": any(row["qnn_has_nan"] for row in rows),
        "any_qnn_inf": any(row["qnn_has_inf"] for row in rows),
        "any_torch_nan": any(row["torch_has_nan"] for row in rows),
        "any_torch_inf": any(row["torch_has_inf"] for row in rows),
        "first_row": rows[0],
    }

    _write_csv(rows, args.csv.expanduser().resolve())
    _write_json(summary, args.json.expanduser().resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

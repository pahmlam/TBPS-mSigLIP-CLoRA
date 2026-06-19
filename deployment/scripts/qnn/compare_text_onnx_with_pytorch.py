#!/usr/bin/env python3
"""Compare an ONNX text encoder (FP32 or QDQ) against the PyTorch baseline.

Text counterpart of `compare_onnx_with_pytorch.py`. The text encoder takes
`input_ids` plus an `attention_mask` (integer or float32, depending on the ONNX
export), both shape [1, 64], produced by `prepare_vn3k_text_inputs.py`. This
reuses those raw files and the same manifest, and calls `TBPS.encode_text({...})`
for the PyTorch reference.

Use it for the same gate as vision: if the QDQ text ONNX already diverges from
PyTorch, the issue is PTQ/calibration (e.g. the 250k embedding table); if QDQ
matches but QNN diverges, debug QNN compile/runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from array import array
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compare_qnn_with_pytorch import (
    PROJECT_ROOT,
    _load_manifest,
    _load_pytorch_model,
    _vector_stats,
    _write_json,
)

_NP_BY_ORT = {
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(float)": np.float32,
}


def _find_single_onnx(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return path
    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one .onnx in {path}, found: {candidates}")
    return candidates[0]


def _parse_dual_input_list(input_dir: Path, input_list_name: str) -> list[dict[str, Path]]:
    """Parse lines of `input_ids:=p1 attention_mask:=p2`."""
    input_list = input_dir / input_list_name
    if not input_list.exists():
        raise FileNotFoundError(f"Input list not found: {input_list}")
    rows: list[dict[str, Path]] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        entry: dict[str, Path] = {}
        for token in line.split():
            if ":=" not in token:
                continue
            name, value = token.split(":=", 1)
            p = Path(value)
            entry[name] = p if p.is_absolute() else input_dir / p
        if entry:
            rows.append(entry)
    if not rows:
        raise ValueError(f"No input rows found in {input_list}")
    return rows


def _read_raw_ints(path: Path, length: int) -> np.ndarray:
    data = path.read_bytes()
    # try int64 then int32 by size
    for dtype in (np.int64, np.int32):
        if len(data) == length * np.dtype(dtype).itemsize:
            return np.frombuffer(data, dtype=dtype).reshape(1, length).copy()
    raise ValueError(f"{path}: {len(data)} bytes does not match {length} int32/int64 values")


def _read_raw_tensor(path: Path, length: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    data = path.read_bytes()
    expected = length * dtype.itemsize
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes does not match {length} {dtype} values")
    return np.frombuffer(data, dtype=dtype).reshape(1, length).copy()


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "result_index", "pid", "caption", "dim", "onnx_norm", "torch_norm",
        "cosine_raw", "cosine_l2", "l2_l2", "onnx_has_nan", "torch_has_nan",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ONNX text encoder vs PyTorch encode_text.")
    p.add_argument("--onnx-model", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--input-list", default="input_list.txt")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--expected-dim", type=int, default=768)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    model_dir = args.model_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    onnx_path = _find_single_onnx(args.onnx_model.expanduser().resolve())
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else input_dir / "manifest.csv"

    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    in_types = {i.name: i.type for i in session.get_inputs()}
    out_name = session.get_outputs()[0].name

    rows_in = _parse_dual_input_list(input_dir, args.input_list)
    manifest_rows = _load_manifest(manifest_path)
    model, _ = _load_pytorch_model(model_dir, args.precision, args.device)

    rows: list[dict] = []
    with torch.no_grad():
        for idx, entry in enumerate(rows_in):
            ids_dtype = _NP_BY_ORT.get(in_types.get("input_ids"), np.int64)
            attn_dtype = _NP_BY_ORT.get(in_types.get("attention_mask"), np.int64)
            ids = _read_raw_ints(entry["input_ids"], args.seq_len)
            if np.dtype(attn_dtype).kind == "f":
                attn = _read_raw_tensor(entry["attention_mask"], args.seq_len, attn_dtype)
            else:
                attn = _read_raw_ints(entry["attention_mask"], args.seq_len)

            caption_input = {
                "input_ids": torch.from_numpy(ids.astype(np.int64)).to(args.device),
                "attention_mask": torch.from_numpy(attn.astype(attn_dtype)).to(args.device),
            }
            torch_out = model.encode_text(caption_input).detach().float().cpu().reshape(-1)

            feed = {
                "input_ids": ids.astype(ids_dtype),
                "attention_mask": attn.astype(attn_dtype),
            }
            onnx_out = torch.from_numpy(
                np.asarray(session.run([out_name], feed)[0], dtype=np.float32)
            ).reshape(-1)

            if onnx_out.numel() != args.expected_dim or torch_out.numel() != args.expected_dim:
                raise SystemExit(
                    f"dim mismatch: onnx={onnx_out.numel()} torch={torch_out.numel()} expected={args.expected_dim}"
                )

            onnx_l2 = F.normalize(onnx_out, dim=0)
            torch_l2 = F.normalize(torch_out, dim=0)
            manifest = manifest_rows[idx] if idx < len(manifest_rows) else {}
            o_stats = _vector_stats(onnx_out.tolist())
            t_stats = _vector_stats(torch_out.tolist())
            rows.append(
                {
                    "result_index": idx,
                    "pid": manifest.get("pid", ""),
                    "caption": (manifest.get("caption", "") or "")[:80],
                    "dim": args.expected_dim,
                    "onnx_norm": o_stats["norm"],
                    "torch_norm": t_stats["norm"],
                    "cosine_raw": F.cosine_similarity(onnx_out.unsqueeze(0), torch_out.unsqueeze(0)).item(),
                    "cosine_l2": torch.dot(onnx_l2, torch_l2).item(),
                    "l2_l2": torch.linalg.vector_norm(onnx_l2 - torch_l2).item(),
                    "onnx_has_nan": o_stats["has_nan"],
                    "torch_has_nan": t_stats["has_nan"],
                }
            )

    cosines = [r["cosine_l2"] for r in rows]
    summary = {
        "onnx_model": str(onnx_path),
        "model_dir": str(model_dir),
        "input_dir": str(input_dir),
        "num_samples": len(rows),
        "expected_dim": args.expected_dim,
        "input_types": in_types,
        "cosine_l2_mean": sum(cosines) / len(cosines),
        "cosine_l2_min": min(cosines),
        "cosine_l2_max": max(cosines),
        "any_onnx_nan": any(r["onnx_has_nan"] for r in rows),
        "first_row": rows[0],
    }
    _write_csv(rows, args.csv.expanduser().resolve())
    _write_json(summary, args.json.expanduser().resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

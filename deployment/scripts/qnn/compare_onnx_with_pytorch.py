#!/usr/bin/env python3
"""Compare an ONNX vision encoder against the local PyTorch baseline.

Use this for the diagnostic step after QAI Hub quantization. If the QDQ ONNX
model already diverges from PyTorch, the issue is PTQ/calibration. If QDQ ONNX
matches PyTorch but QNN output diverges, debug QNN compile/runtime/I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compare_qnn_with_pytorch import (
    PROJECT_ROOT,
    _load_manifest,
    _load_pytorch_model,
    _parse_input_list,
    _read_raw_tensor,
    _vector_stats,
    _write_json,
)


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


def _numpy_for_input(image: torch.Tensor, input_type: str) -> np.ndarray:
    value = image.cpu().numpy()
    if input_type == "tensor(float)":
        return value.astype(np.float32, copy=False)
    if input_type == "tensor(float16)":
        return value.astype(np.float16, copy=False)
    raise SystemExit(
        f"ONNX input type {input_type!r} is not floating-point. This compare "
        "script expects a QDQ ONNX model with float I/O; native INT I/O needs "
        "the model's quantization encoding to prepare inputs correctly."
    )


def _ensure_float_output(values: np.ndarray, output_type: str) -> torch.Tensor:
    if output_type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
        raise SystemExit(
            f"ONNX output type {output_type!r} is not floating-point. Compare "
            "the dequantized output or provide output encodings first."
        )
    return torch.from_numpy(values.astype(np.float32, copy=False)).reshape(-1)


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "result_index",
        "pid",
        "dataset_file_path",
        "raw_path",
        "dim",
        "onnx_norm",
        "torch_norm",
        "cosine_raw",
        "cosine_l2",
        "l2_raw",
        "l2_l2",
        "mean_abs_raw",
        "max_abs_raw",
        "onnx_has_nan",
        "onnx_has_inf",
        "torch_has_nan",
        "torch_has_inf",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an ONNX vision encoder with PyTorch encode_image outputs."
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        required=True,
        help="ONNX model file or directory containing one .onnx file.",
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
        default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_10",
        help="Input directory containing input_list.txt and raw files.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional manifest.csv path.")
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--output-name", help="Optional ONNX output tensor name.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--rotated",
        action="store_true",
        help="Load --model-dir's PyTorch reference via load_rotated_model "
        "(LayerNorm->RMSNorm swap before loading folded weights). Required when "
        "the model-dir is a rotated export; plain _load_pytorch_model gives ~0.11.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_outputs/vn3k_test_10/onnx_vs_pytorch.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_outputs/vn3k_test_10/onnx_vs_pytorch_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import onnxruntime as ort

    model_dir = args.model_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    onnx_path = _find_single_onnx(args.onnx_model.expanduser().resolve())
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else input_dir / "manifest.csv"
    )

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    output_meta = (
        next(output for output in session.get_outputs() if output.name == args.output_name)
        if args.output_name
        else session.get_outputs()[0]
    )

    raw_paths = _parse_input_list(input_dir, args.input_list)
    manifest_rows = _load_manifest(manifest_path)
    if args.rotated:
        from rotate_vision_encoder import load_rotated_model

        model, _ = load_rotated_model(model_dir, args.device)
    else:
        model, _ = _load_pytorch_model(model_dir, args.precision, args.device)

    rows: list[dict] = []
    with torch.no_grad():
        for idx, raw_path in enumerate(raw_paths):
            image = _read_raw_tensor(raw_path, args.image_size).to(args.device)
            if args.precision == "fp16":
                image = image.half()

            torch_output = model.encode_image(image).detach().float().cpu().reshape(-1)
            onnx_values = session.run(
                [output_meta.name],
                {input_meta.name: _numpy_for_input(image.float(), input_meta.type)},
            )[0]
            onnx_output = _ensure_float_output(onnx_values, output_meta.type)

            if onnx_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"ONNX output for {raw_path} has dim={onnx_output.numel()}, "
                    f"expected {args.expected_dim}"
                )
            if torch_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"PyTorch output for {raw_path} has dim={torch_output.numel()}, "
                    f"expected {args.expected_dim}"
                )

            diff = onnx_output - torch_output
            onnx_l2 = F.normalize(onnx_output, dim=0)
            torch_l2 = F.normalize(torch_output, dim=0)
            l2_diff = onnx_l2 - torch_l2
            manifest = manifest_rows[idx] if idx < len(manifest_rows) else {}
            onnx_stats = _vector_stats(onnx_output.tolist())
            torch_stats = _vector_stats(torch_output.tolist())

            rows.append(
                {
                    "result_index": idx,
                    "pid": manifest.get("pid", ""),
                    "dataset_file_path": manifest.get("dataset_file_path", ""),
                    "raw_path": str(raw_path),
                    "dim": args.expected_dim,
                    "onnx_norm": onnx_stats["norm"],
                    "torch_norm": torch_stats["norm"],
                    "cosine_raw": F.cosine_similarity(
                        onnx_output.unsqueeze(0), torch_output.unsqueeze(0)
                    ).item(),
                    "cosine_l2": torch.dot(onnx_l2, torch_l2).item(),
                    "l2_raw": torch.linalg.vector_norm(diff).item(),
                    "l2_l2": torch.linalg.vector_norm(l2_diff).item(),
                    "mean_abs_raw": torch.mean(torch.abs(diff)).item(),
                    "max_abs_raw": torch.max(torch.abs(diff)).item(),
                    "onnx_has_nan": onnx_stats["has_nan"],
                    "onnx_has_inf": onnx_stats["has_inf"],
                    "torch_has_nan": torch_stats["has_nan"],
                    "torch_has_inf": torch_stats["has_inf"],
                }
            )

    cosines = [row["cosine_l2"] for row in rows]
    l2_l2 = [row["l2_l2"] for row in rows]
    mean_abs = [row["mean_abs_raw"] for row in rows]
    max_abs = [row["max_abs_raw"] for row in rows]
    summary = {
        "onnx_model": str(onnx_path),
        "model_dir": str(model_dir),
        "input_dir": str(input_dir),
        "precision": args.precision,
        "device": args.device,
        "num_samples": len(rows),
        "expected_dim": args.expected_dim,
        "input_name": input_meta.name,
        "input_type": input_meta.type,
        "output_name": output_meta.name,
        "output_type": output_meta.type,
        "cosine_l2_mean": sum(cosines) / len(cosines),
        "cosine_l2_min": min(cosines),
        "cosine_l2_max": max(cosines),
        "l2_l2_mean": sum(l2_l2) / len(l2_l2),
        "l2_l2_max": max(l2_l2),
        "mean_abs_raw_mean": sum(mean_abs) / len(mean_abs),
        "max_abs_raw_max": max(max_abs),
        "any_onnx_nan": any(row["onnx_has_nan"] for row in rows),
        "any_onnx_inf": any(row["onnx_has_inf"] for row in rows),
        "any_torch_nan": any(row["torch_has_nan"] for row in rows),
        "any_torch_inf": any(row["torch_has_inf"] for row in rows),
        "first_row": rows[0],
    }

    _write_csv(rows, args.csv.expanduser().resolve())
    _write_json(summary, args.json.expanduser().resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

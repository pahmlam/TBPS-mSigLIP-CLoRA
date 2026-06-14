#!/usr/bin/env python3
"""Compare QNN vision outputs against a local PyTorch baseline.

The comparison reuses the exact float32 raw image tensors passed to qnn-net-run,
so the measured difference is from model/runtime/quantization, not JPEG decoding
or preprocessing drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
from array import array
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _read_float32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not divisible by 4 bytes: {len(data)}")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def _read_raw_tensor(path: Path, image_size: int):
    import torch

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
    return torch.tensor(values, dtype=torch.float32).view(1, 3, image_size, image_size)


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


def _result_index(path: Path) -> int:
    try:
        return int(path.parent.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _load_manifest(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _register_omegaconf_resolvers() -> None:
    from omegaconf import OmegaConf

    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_pytorch_model(model_dir: Path, precision: str, device: str):
    import torch
    import yaml
    from omegaconf import OmegaConf

    from msiglip.model.build import build_backbone_with_proper_layer_resize
    from msiglip.model.tbps import TBPS

    _register_omegaconf_resolvers()

    config_path = model_dir / "config.yaml"
    state_path = model_dir / f"model_{precision}.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State dict not found: {state_path}")

    with config_path.open(encoding="utf-8") as f:
        config = OmegaConf.create(yaml.safe_load(f))

    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    backbone = build_backbone_with_proper_layer_resize(config.backbone)
    model = TBPS(config=config, backbone=backbone)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: unexpected state_dict keys: {len(unexpected)}")
    if missing:
        print(f"Warning: missing state_dict keys: {len(missing)}")

    model.eval()
    if precision == "fp16":
        model = model.half()
    model = model.to(device)
    return model, config


def _vector_stats(values: list[float]) -> dict:
    has_nan = any(math.isnan(value) for value in values)
    has_inf = any(math.isinf(value) for value in values)
    if has_nan or has_inf:
        return {
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "norm": math.nan,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "finite_count": sum(math.isfinite(value) for value in values),
        }

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values),
        "norm": math.sqrt(sum(value * value for value in values)),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "finite_count": len(values),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "result_index",
        "pid",
        "dataset_file_path",
        "raw_path",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare QNN Result_*/output_0.raw with PyTorch encode_image outputs."
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
        help="QNN input directory containing input_list.txt and raw files.",
    )
    parser.add_argument(
        "--qnn-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_outputs/vn3k_test_10",
        help="QNN output directory containing Result_*/output_0.raw.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional manifest.csv path.")
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--output-name", default="output_0.raw")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn.functional as F

    model_dir = args.model_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    qnn_output_dir = args.qnn_output_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else input_dir / "manifest.csv"
    )

    raw_paths = _parse_input_list(input_dir, args.input_list)
    qnn_paths = sorted(
        qnn_output_dir.glob(f"Result_*/{args.output_name}"), key=_result_index
    )
    if len(raw_paths) != len(qnn_paths):
        raise SystemExit(
            f"Input/output count mismatch: {len(raw_paths)} raw inputs vs "
            f"{len(qnn_paths)} QNN outputs"
        )

    manifest_rows = _load_manifest(manifest_path)
    model, _ = _load_pytorch_model(model_dir, args.precision, args.device)

    rows: list[dict] = []
    with torch.no_grad():
        for idx, (raw_path, qnn_path) in enumerate(zip(raw_paths, qnn_paths)):
            image = _read_raw_tensor(raw_path, args.image_size).to(args.device)
            if args.precision == "fp16":
                image = image.half()

            torch_output = model.encode_image(image).detach().float().cpu().reshape(-1)
            qnn_output = torch.tensor(_read_float32(qnn_path), dtype=torch.float32)

            if qnn_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"{qnn_path} has dim={qnn_output.numel()}, expected {args.expected_dim}"
                )
            if torch_output.numel() != args.expected_dim:
                raise SystemExit(
                    f"PyTorch output for {raw_path} has dim={torch_output.numel()}, "
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
                    "dataset_file_path": manifest.get("dataset_file_path", ""),
                    "raw_path": str(raw_path),
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

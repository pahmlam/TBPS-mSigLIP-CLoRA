#!/usr/bin/env python3
"""Evaluate VN3K retrieval with board-generated vision embeddings.

Use this after running the vision context binary on RB3 with the full VN3K test
gallery. The script reads qnn-net-run `Result_*/output_0.raw` files, maps them
to pids through the prepared gallery `manifest.csv`, encodes text captions with
the FP32 PyTorch model, and reports the same raw-dot-product R@1 metric used by
the training evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
QNN_SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, QNN_SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_qnn_with_pytorch import (  # noqa: E402
    _load_pytorch_model,
    _read_float32,
    _register_omegaconf_resolvers,
    _result_index,
)
from eval_retrieval_quantized_vision import (  # noqa: E402
    DEFAULT_GATE_R1,
    FP32_BASELINE_R1,
    _build_test_sets,
    _encode_text,
    _load_config,
    _metrics,
    _print_table,
)
from msiglip.utils.tokenizer_utils import get_tokenizer  # noqa: E402


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Gallery manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_board_image_embeddings(
    output_dir: Path,
    gallery_input_dir: Path,
    output_name: str,
    expected_dim: int,
    max_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    manifest_rows = _read_manifest(gallery_input_dir / "manifest.csv")
    qnn_paths = sorted(output_dir.glob(f"Result_*/{output_name}"), key=_result_index)
    if not qnn_paths:
        raise FileNotFoundError(f"No Result_*/{output_name} files found under {output_dir}")
    if len(qnn_paths) > len(manifest_rows):
        raise ValueError(
            f"Found {len(qnn_paths)} QNN outputs but only {len(manifest_rows)} manifest rows"
        )

    if max_images:
        qnn_paths = qnn_paths[:max_images]

    feats: list[torch.Tensor] = []
    pids: list[int] = []
    for qnn_path in qnn_paths:
        idx = _result_index(qnn_path)
        if idx < 0 or idx >= len(manifest_rows):
            raise ValueError(f"Cannot map {qnn_path} to manifest row index {idx}")
        values = torch.tensor(_read_float32(qnn_path), dtype=torch.float32)
        if values.numel() != expected_dim:
            raise ValueError(f"{qnn_path} has dim={values.numel()}, expected {expected_dim}")
        feats.append(values.reshape(1, -1))
        pids.append(int(manifest_rows[idx]["pid"]))

    return torch.cat(feats, dim=0), torch.tensor(pids, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vision-output-dir",
        type=Path,
        required=True,
        help="Board qnn-net-run output dir containing Result_*/output_0.raw.",
    )
    parser.add_argument(
        "--gallery-input-dir",
        type=Path,
        required=True,
        help="Prepared VN3K gallery input dir containing manifest.csv.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model",
        help="Merged FP32 model dir used to encode text.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing VN3K/. Default: project root.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--output-name", default="output_0.raw")
    parser.add_argument("--max-images", type=int, default=0, help="Smoke subset only.")
    parser.add_argument("--max-captions", type=int, default=0, help="Smoke subset only.")
    parser.add_argument("--gate-r1", type=float, default=DEFAULT_GATE_R1)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/board_vision_r1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_omegaconf_resolvers()

    model_dir = args.model_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    vision_output_dir = args.vision_output_dir.expanduser().resolve()
    gallery_input_dir = args.gallery_input_dir.expanduser().resolve()

    config = _load_config(model_dir)
    tokenizer = get_tokenizer(config.tokenizer)
    _, test_txt_set = _build_test_sets(config, dataset_root, tokenizer)

    device = torch.device(args.device)
    model, _ = _load_pytorch_model(model_dir, "fp32", str(device))

    txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    text_fp32, _, text_ids, _ = _encode_text(
        model, txt_loader, device, args.max_captions, session=None
    )
    image_board, image_ids = _load_board_image_embeddings(
        vision_output_dir,
        gallery_input_dir,
        args.output_name,
        args.expected_dim,
        args.max_images,
    )

    print(
        f"Embeddings: text FP32 {tuple(text_fp32.shape)}, "
        f"image board {tuple(image_board.shape)}"
    )
    metrics = _metrics(text_fp32, image_board, text_ids, image_ids, normalize=False)
    _print_table("BOARD VISION-INT8 (text FP32 + image QNN board)", metrics)

    subset = bool(args.max_images or args.max_captions)
    gate_pass = metrics["t2i"]["R1"] >= args.gate_r1
    verdict = "PASS" if gate_pass else "FAIL"
    if subset:
        verdict += " (SUBSET SMOKE - not the real gate)"
    print(f"\nBOARD VISION GATE T2I R@1 >= {args.gate_r1}: {metrics['t2i']['R1']:.2f} -> {verdict}")

    summary = {
        "vision_output_dir": str(vision_output_dir),
        "gallery_input_dir": str(gallery_input_dir),
        "model_dir": str(model_dir),
        "num_gallery_images": int(image_ids.numel()),
        "num_query_captions": int(text_ids.numel()),
        "subset_smoke": subset,
        "similarity": "raw_dot_product",
        "combo": "board_vision_int8_text_fp32",
        "metrics": metrics,
        "gate_r1": args.gate_r1,
        "gate_pass": gate_pass,
        "baseline_reference_t2i_r1": FP32_BASELINE_R1,
    }

    out = args.json.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

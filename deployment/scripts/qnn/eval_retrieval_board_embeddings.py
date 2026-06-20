#!/usr/bin/env python3
"""Evaluate VN3K retrieval with board-generated image and text embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
QNN_SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, QNN_SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_qnn_with_pytorch import _register_omegaconf_resolvers  # noqa: E402
from eval_retrieval_board_text import _load_board_text_embeddings  # noqa: E402
from eval_retrieval_board_vision import _load_board_image_embeddings  # noqa: E402
from eval_retrieval_quantized_vision import (  # noqa: E402
    DEFAULT_GATE_R1,
    FP32_BASELINE_R1,
    _metrics,
    _print_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vision-output-dir",
        type=Path,
        required=True,
        help="Board vision output dir containing Result_*/output_0.raw.",
    )
    parser.add_argument(
        "--text-output-dir",
        type=Path,
        required=True,
        help="Board text output dir containing Result_*/output_0.raw.",
    )
    parser.add_argument(
        "--gallery-input-dir",
        type=Path,
        required=True,
        help="Prepared VN3K gallery input dir containing manifest.csv.",
    )
    parser.add_argument(
        "--query-input-dir",
        type=Path,
        required=True,
        help="Prepared VN3K text query input dir containing manifest.csv.",
    )
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--output-name", default="output_0.raw")
    parser.add_argument("--max-images", type=int, default=0, help="Smoke subset only.")
    parser.add_argument("--max-captions", type=int, default=0, help="Smoke subset only.")
    parser.add_argument("--gate-r1", type=float, default=DEFAULT_GATE_R1)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/qnn_runs/both_int8_board_r1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_omegaconf_resolvers()

    vision_output_dir = args.vision_output_dir.expanduser().resolve()
    text_output_dir = args.text_output_dir.expanduser().resolve()
    gallery_input_dir = args.gallery_input_dir.expanduser().resolve()
    query_input_dir = args.query_input_dir.expanduser().resolve()

    image_board, image_ids = _load_board_image_embeddings(
        vision_output_dir,
        gallery_input_dir,
        args.output_name,
        args.expected_dim,
        args.max_images,
    )
    text_board, text_ids = _load_board_text_embeddings(
        text_output_dir,
        query_input_dir,
        args.output_name,
        args.expected_dim,
        args.max_captions,
    )

    print(
        f"Embeddings: text board {tuple(text_board.shape)}, "
        f"image board {tuple(image_board.shape)}"
    )
    metrics = _metrics(text_board, image_board, text_ids, image_ids, normalize=False)
    _print_table("BOARD BOTH-INT8 (image QNN board + text QNN board)", metrics)

    subset = bool(args.max_images or args.max_captions)
    gate_pass = metrics["t2i"]["R1"] >= args.gate_r1
    verdict = "PASS" if gate_pass else "FAIL"
    if subset:
        verdict += " (SUBSET SMOKE - not the real gate)"
    print(f"\nBOARD BOTH-INT8 GATE T2I R@1 >= {args.gate_r1}: {metrics['t2i']['R1']:.2f} -> {verdict}")

    summary = {
        "vision_output_dir": str(vision_output_dir),
        "text_output_dir": str(text_output_dir),
        "gallery_input_dir": str(gallery_input_dir),
        "query_input_dir": str(query_input_dir),
        "num_gallery_images": int(image_ids.numel()),
        "num_query_captions": int(text_ids.numel()),
        "subset_smoke": subset,
        "similarity": "raw_dot_product",
        "combo": "board_both_int8",
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

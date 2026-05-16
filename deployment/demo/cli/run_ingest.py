"""CLI: source -> detect -> track -> crop -> image embedding -> spool/store."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .builders import build_ingest_components
from ..core.pipeline import IngestPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular mSigLIP ingest demo.")
    parser.add_argument("--source", required=True, help="Image file, image directory, or video file.")
    parser.add_argument("--source-type", choices=["auto", "images", "video"], default="auto")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--board-id", default="local-dev")
    parser.add_argument("--camera-id", default="cam-local-01")
    parser.add_argument("--model-version", default="msiglip-demo-wiring")
    parser.add_argument("--store", default="deployment/demo_runtime/vectors.jsonl")
    parser.add_argument("--spool", default="deployment/demo_runtime/spool")
    parser.add_argument("--crops", default="deployment/demo_runtime/crops")
    parser.add_argument("--upload-mode", choices=["local", "fail"], default="local")
    parser.add_argument("--tracker-mode", choices=["per_frame", "single"], default="per_frame")

    parser.add_argument("--encoder", choices=["fake", "onnx", "qnn"], default="fake")
    parser.add_argument("--onnx-model", default="exported_model/vision_onnx/vision_encoder.onnx")
    parser.add_argument("--vision-bin", default="vision_encoder.bin")
    parser.add_argument("--htp-config", default="htp_config_245.json")
    parser.add_argument("--qairt", default="/opt/qcom/qairt/2.45.40.260406")
    parser.add_argument("--qnn-bin")
    parser.add_argument("--qnn-lib")
    parser.add_argument("--runtime-dir", default="deployment/demo_runtime/qnn")
    parser.add_argument("--adsp-library-path")
    parser.add_argument("--keep-qnn-artifacts", action="store_true")

    parser.add_argument("--max-snapshots-per-track", type=int, default=3)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.03)
    parser.add_argument("--min-quality-score", type=float, default=0.05)
    parser.add_argument("--min-quality-improvement", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = build_ingest_components(args)
    pipeline = IngestPipeline(
        **components,
        crop_dir=Path(args.crops),
        board_id=args.board_id,
        camera_id=args.camera_id,
        model_version=args.model_version,
    )
    stats = pipeline.run()
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()

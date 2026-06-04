"""Factory helpers for demo CLIs."""

from __future__ import annotations

from pathlib import Path

from ..adapters.crop_selector import DefaultCropSelector
from ..adapters.detectors import FullFramePersonDetector
from ..adapters.encoders import FakeTextEncoder, FakeVisionEncoder, OnnxVisionEncoder, QnnVisionEncoder
from ..adapters.sources import build_source
from ..adapters.spool import DiskSpool
from ..adapters.tracker import SimpleTracker
from ..adapters.uploaders import FailingUploader, HttpUploader, LocalVectorStoreUploader
from ..adapters.vector_store import JsonlVectorStore


def build_vision_encoder(args):
    if args.encoder == "fake":
        return FakeVisionEncoder()
    if args.encoder == "onnx":
        return OnnxVisionEncoder(Path(args.onnx_model))
    if args.encoder == "qnn":
        return QnnVisionEncoder(
            vision_bin=Path(args.vision_bin),
            htp_config=Path(args.htp_config),
            qairt=Path(args.qairt),
            qnn_bin=Path(args.qnn_bin) if args.qnn_bin else None,
            qnn_lib=Path(args.qnn_lib) if args.qnn_lib else None,
            runtime_dir=Path(args.runtime_dir),
            keep_artifacts=args.keep_qnn_artifacts,
            adsp_library_path=Path(args.adsp_library_path) if args.adsp_library_path else None,
        )
    raise ValueError(f"Unknown encoder: {args.encoder}")


def build_text_encoder(args):
    if args.text_encoder == "fake":
        return FakeTextEncoder()
    raise ValueError(f"Unknown text encoder: {args.text_encoder}")


def build_ingest_components(args):
    vector_store = JsonlVectorStore(Path(args.store))
    if args.upload_mode == "local":
        uploader = LocalVectorStoreUploader(vector_store)
    elif args.upload_mode == "http":
        if not args.backend_url:
            raise ValueError("--backend-url is required with --upload-mode http")
        uploader = HttpUploader(
            backend_url=args.backend_url,
            board_token=args.board_token,
            timeout=args.http_timeout,
        )
    else:
        uploader = FailingUploader()
    return {
        "source": build_source(Path(args.source), args.source_type, max_frames=args.max_frames),
        "detector": FullFramePersonDetector(),
        "tracker": SimpleTracker(mode=args.tracker_mode),
        "crop_selector": DefaultCropSelector(
            max_snapshots_per_track=args.max_snapshots_per_track,
            min_bbox_area_ratio=args.min_bbox_area_ratio,
            min_quality_score=args.min_quality_score,
            min_quality_improvement=args.min_quality_improvement,
        ),
        "image_encoder": build_vision_encoder(args),
        "spool": DiskSpool(Path(args.spool)),
        "uploader": uploader,
    }

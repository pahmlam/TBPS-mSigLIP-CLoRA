"""Pipeline orchestration for the RB3-first modular demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import (
    CropSelector,
    EmbeddingRecord,
    FrameSource,
    ImageEncoder,
    PersonDetector,
    Spool,
    Tracker,
    Uploader,
)
from .utils import ensure_dir, stable_id, to_jsonable


@dataclass
class IngestStats:
    frames_seen: int = 0
    detections_seen: int = 0
    crops_selected: int = 0
    embeddings_written: int = 0
    uploads_failed: int = 0


class IngestPipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: PersonDetector,
        tracker: Tracker,
        crop_selector: CropSelector,
        image_encoder: ImageEncoder,
        spool: Spool,
        uploader: Uploader,
        crop_dir: Path,
        board_id: str,
        camera_id: str,
        model_version: str,
    ):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.crop_selector = crop_selector
        self.image_encoder = image_encoder
        self.spool = spool
        self.uploader = uploader
        self.crop_dir = ensure_dir(crop_dir.expanduser().resolve())
        self.board_id = board_id
        self.camera_id = camera_id
        self.model_version = model_version

    def _crop_path(self, candidate_index: int, record_id: str) -> Path:
        return self.crop_dir / f"{candidate_index:06d}_{record_id}.jpg"

    def _record_id(self, candidate) -> str:
        return stable_id(
            self.board_id,
            self.camera_id,
            candidate.track.track_id,
            candidate.track.frame.frame_id,
            candidate.captured_at,
            candidate.bbox,
        )

    def _record(self, candidate, embedding: list[float], crop_path: Path) -> EmbeddingRecord:
        record_id = self._record_id(candidate)
        return EmbeddingRecord(
            id=record_id,
            board_id=self.board_id,
            camera_id=self.camera_id,
            track_id=candidate.track.track_id,
            episode_id=candidate.track.episode_id,
            captured_at=candidate.captured_at,
            bbox=candidate.bbox,
            quality_score=float(candidate.quality_score),
            embedding=embedding,
            crop_path=str(crop_path),
            model_version=self.model_version,
            save_reason=candidate.save_reason,
            metadata={
                "source_path": candidate.track.frame.source_path,
                "frame_id": candidate.track.frame.frame_id,
                "encoder_runtime": self.image_encoder.runtime_name,
                "first_seen": candidate.track.first_seen,
                "last_seen": candidate.track.last_seen,
                "num_frames_seen": candidate.track.num_frames_seen,
            },
        )

    def run(self) -> IngestStats:
        stats = IngestStats()
        crop_index = 0
        for frame in self.source.frames():
            stats.frames_seen += 1
            detections = self.detector.detect(frame)
            stats.detections_seen += len(detections)
            observations = self.tracker.update(frame, detections)
            candidates = self.crop_selector.select(observations)
            stats.crops_selected += len(candidates)

            for candidate in candidates:
                embedding = self.image_encoder.encode(candidate.crop)
                record_id = self._record_id(candidate)
                crop_path = self._crop_path(crop_index, record_id)
                candidate.crop.save(crop_path, format="JPEG", quality=90)
                record = self._record(candidate, embedding, crop_path)
                event = {
                    "type": "track_embedding",
                    "record": asdict(record),
                }
                spool_id = self.spool.enqueue(event)
                result = self.uploader.upload(to_jsonable(event))
                if result.ok:
                    self.spool.mark_sent(spool_id, result)
                    stats.embeddings_written += 1
                else:
                    self.spool.mark_failed(spool_id, result)
                    stats.uploads_failed += 1
                crop_index += 1
        return stats

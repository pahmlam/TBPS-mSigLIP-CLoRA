"""Shared data contracts and plugin protocols for the deployment demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Frame:
    frame_id: str
    image: Any
    timestamp: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    label: str = "person"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    episode_id: str
    frame: Frame
    detection: Detection
    first_seen: str
    last_seen: str
    num_frames_seen: int


@dataclass(frozen=True)
class CropCandidate:
    track: TrackObservation
    crop: Any
    bbox: BBox
    quality_score: float
    save_reason: str
    captured_at: str


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    board_id: str
    camera_id: str
    track_id: str
    episode_id: str
    captured_at: str
    bbox: BBox
    quality_score: float
    embedding: list[float]
    crop_path: str
    model_version: str
    save_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    record: EmbeddingRecord
    score: float


@dataclass(frozen=True)
class UploadResult:
    ok: bool
    remote_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class HealthSnapshot:
    board_id: str
    camera_id: str
    encoder_runtime: str
    upload_backlog: int
    spool_pending: int
    spool_sent: int
    spool_failed: int
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameSource(Protocol):
    def frames(self) -> Iterable[Frame]:
        """Yield decoded RGB frames."""


class PersonDetector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]:
        """Return person detections for a frame."""


class Tracker(Protocol):
    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]:
        """Assign track and episode ids to detections."""


class CropSelector(Protocol):
    def select(self, observations: list[TrackObservation]) -> list[CropCandidate]:
        """Choose which observations should be embedded."""


class ImageEncoder(Protocol):
    runtime_name: str

    def encode(self, crop: Any) -> list[float]:
        """Return a finite L2-normalized 768-d image embedding."""


class TextEncoder(Protocol):
    runtime_name: str

    def encode(self, text: str) -> list[float]:
        """Return a finite L2-normalized 768-d text embedding."""


class VectorStore(Protocol):
    def upsert(self, record: EmbeddingRecord) -> None:
        """Store or append an embedding record."""

    def search(
        self,
        query_embedding: list[float],
        top_k_raw: int,
        top_k_final: int,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        collapse_key: str = "episode_id",
    ) -> list[SearchResult]:
        """Search vectors with cosine similarity and optional metadata filters."""


class Spool(Protocol):
    def enqueue(self, event: dict[str, Any]) -> str:
        """Persist a pending event and return its spool id."""

    def mark_sent(self, spool_id: str, result: UploadResult) -> None:
        """Move a pending event to sent state."""

    def mark_failed(self, spool_id: str, result: UploadResult) -> None:
        """Move a pending event to failed state."""


class Uploader(Protocol):
    def upload(self, event: dict[str, Any]) -> UploadResult:
        """Upload or otherwise commit an event."""

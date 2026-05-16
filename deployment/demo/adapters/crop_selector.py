"""Crop selection and lightweight suppression policy."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.contracts import CropCandidate, TrackObservation


@dataclass
class _TrackSaveState:
    saved_count: int = 0
    best_quality: float = 0.0


class DefaultCropSelector:
    """Select up to N good crops per track.

    This implements the first deployable suppression layer from the design doc:
    do not embed every frame, keep a small number of representative snapshots.
    """

    def __init__(
        self,
        max_snapshots_per_track: int = 3,
        min_bbox_area_ratio: float = 0.03,
        min_quality_score: float = 0.05,
        min_quality_improvement: float = 0.08,
    ):
        self.max_snapshots_per_track = max_snapshots_per_track
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.min_quality_score = min_quality_score
        self.min_quality_improvement = min_quality_improvement
        self._states: dict[str, _TrackSaveState] = {}

    @staticmethod
    def _bbox_area_ratio(obs: TrackObservation) -> float:
        x1, y1, x2, y2 = obs.detection.bbox
        width, height = obs.frame.image.size
        area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = max(1, width * height)
        return area / frame_area

    def _quality(self, obs: TrackObservation) -> float:
        area_ratio = self._bbox_area_ratio(obs)
        return max(0.0, min(1.0, obs.detection.confidence * area_ratio))

    def select(self, observations: list[TrackObservation]) -> list[CropCandidate]:
        selected: list[CropCandidate] = []
        for obs in observations:
            area_ratio = self._bbox_area_ratio(obs)
            quality = self._quality(obs)
            if area_ratio < self.min_bbox_area_ratio or quality < self.min_quality_score:
                continue

            state = self._states.setdefault(obs.track_id, _TrackSaveState())
            if state.saved_count >= self.max_snapshots_per_track:
                continue
            if state.saved_count > 0 and quality < state.best_quality + self.min_quality_improvement:
                continue

            x1, y1, x2, y2 = obs.detection.bbox
            crop = obs.frame.image.crop((x1, y1, x2, y2))
            reason = "best_frame" if state.saved_count == 0 else "quality_improvement"
            state.saved_count += 1
            state.best_quality = max(state.best_quality, quality)
            selected.append(
                CropCandidate(
                    track=obs,
                    crop=crop,
                    bbox=obs.detection.bbox,
                    quality_score=quality,
                    save_reason=reason,
                    captured_at=obs.frame.timestamp,
                )
            )
        return selected

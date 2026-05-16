"""Simple tracker adapter for the first modular demo."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.contracts import Detection, Frame, TrackObservation


@dataclass
class _TrackState:
    first_seen: str
    last_seen: str
    num_frames_seen: int = 0


class SimpleTracker:
    """Assign deterministic track and episode ids.

    `per_frame` is best for image-directory preflight where each image is an
    independent crop. `single` is useful for a short one-person video smoke test.
    """

    def __init__(self, mode: str = "per_frame"):
        if mode not in {"per_frame", "single"}:
            raise ValueError("tracker mode must be 'per_frame' or 'single'")
        self.mode = mode
        self._states: dict[str, _TrackState] = {}

    def _track_id(self, frame: Frame, detection_index: int) -> str:
        if self.mode == "single":
            return f"trk-{detection_index:04d}"
        return f"trk-{frame.frame_id}-{detection_index:04d}"

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]:
        observations: list[TrackObservation] = []
        for detection_index, detection in enumerate(detections):
            track_id = self._track_id(frame, detection_index)
            state = self._states.get(track_id)
            if state is None:
                state = _TrackState(first_seen=frame.timestamp, last_seen=frame.timestamp)
                self._states[track_id] = state
            state.last_seen = frame.timestamp
            state.num_frames_seen += 1
            episode_id = track_id.replace("trk-", "eps-", 1)
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    episode_id=episode_id,
                    frame=frame,
                    detection=detection,
                    first_seen=state.first_seen,
                    last_seen=state.last_seen,
                    num_frames_seen=state.num_frames_seen,
                )
            )
        return observations

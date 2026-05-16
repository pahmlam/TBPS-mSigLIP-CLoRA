"""Person detector adapters."""

from __future__ import annotations

from ..core.contracts import Detection, Frame


class FullFramePersonDetector:
    """Treat the whole image as a person crop.

    This is the v1 adapter for VN3K-style cropped person images and local wiring
    checks. A YOLO/ByteTrack stack can replace it later behind the same protocol.
    """

    def __init__(self, confidence: float = 1.0):
        self.confidence = confidence

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = frame.image.size
        return [
            Detection(
                bbox=(0, 0, int(width), int(height)),
                confidence=self.confidence,
                label="person",
                metadata={"detector": "full_frame"},
            )
        ]

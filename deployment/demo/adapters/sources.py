"""Frame source adapters for the modular deployment demo."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.contracts import Frame
from ..core.utils import utc_now_iso


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDirectorySource:
    """Yield frames from an image file or a directory of images."""

    def __init__(self, path: Path, max_frames: int | None = None):
        self.path = path.expanduser().resolve()
        self.max_frames = max_frames

    def _paths(self) -> list[Path]:
        if self.path.is_file():
            if self.path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {self.path}")
            return [self.path]
        if not self.path.is_dir():
            raise FileNotFoundError(f"Image source not found: {self.path}")
        paths = [
            path
            for path in sorted(self.path.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            raise ValueError(f"No supported images found under {self.path}")
        return paths

    def frames(self) -> Iterable[Frame]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Pillow is required for image sources") from exc

        for index, path in enumerate(self._paths()):
            if self.max_frames is not None and index >= self.max_frames:
                break
            with Image.open(path) as img:
                yield Frame(
                    frame_id=f"img-{index:06d}",
                    image=img.convert("RGB").copy(),
                    timestamp=utc_now_iso(),
                    source_path=str(path),
                    metadata={"source_type": "image"},
                )


class VideoFileSource:
    """Yield frames from a video file through OpenCV when available."""

    def __init__(self, path: Path, max_frames: int | None = None, stride: int = 1):
        self.path = path.expanduser().resolve()
        self.max_frames = max_frames
        self.stride = max(1, stride)

    def frames(self) -> Iterable[Frame]:
        if not self.path.exists():
            raise FileNotFoundError(f"Video source not found: {self.path}")
        try:
            import cv2
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "VideoFileSource requires opencv-python and Pillow. "
                "Use ImageDirectorySource for dependency-light preflight."
            ) from exc

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")
        yielded = 0
        index = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if index % self.stride != 0:
                    index += 1
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield Frame(
                    frame_id=f"vid-{index:06d}",
                    image=Image.fromarray(frame_rgb),
                    timestamp=utc_now_iso(),
                    source_path=str(self.path),
                    metadata={"source_type": "video", "video_frame_index": index},
                )
                yielded += 1
                index += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
        finally:
            cap.release()


def build_source(source: Path, source_type: str = "auto", max_frames: int | None = None):
    source = source.expanduser().resolve()
    resolved_type = source_type
    if source_type == "auto":
        if source.is_dir() or source.suffix.lower() in IMAGE_EXTENSIONS:
            resolved_type = "images"
        else:
            resolved_type = "video"
    if resolved_type == "images":
        return ImageDirectorySource(source, max_frames=max_frames)
    if resolved_type == "video":
        return VideoFileSource(source, max_frames=max_frames)
    raise ValueError(f"Unknown source type: {source_type}")

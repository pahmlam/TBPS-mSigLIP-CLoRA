"""Small utilities shared by demo adapters."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_id(*parts: object, length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:length]


def l2_normalize(values: list[float], expected_dim: int = 768) -> list[float]:
    if len(values) != expected_dim:
        raise ValueError(f"Embedding dim={len(values)}, expected {expected_dim}")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding contains NaN or Inf")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("Embedding norm is zero")
    return [value / norm for value in values]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimensions differ: {len(a)} vs {len(b)}")
    return float(sum(x * y for x, y in zip(a, b)))


def read_float32_raw(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not divisible by 4 bytes: {len(data)}")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def write_msiglip_image_raw(image: Any, raw_path: Path, image_size: int = 256) -> None:
    """Write RGB image as NCHW float32 raw using mSigLIP preprocessing.

    Preprocessing matches deployment/scripts/qnn/prepare_vn3k_vision_inputs.py:
    RGB -> resize 256x256 bicubic -> ToTensor [0,1] -> Normalize(0.5,0.5).
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for image preprocessing") from exc

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image for image preprocessing")

    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # Pillow < 9
        resample = Image.BICUBIC

    img = image.convert("RGB").resize((image_size, image_size), resample)
    pixels = list(img.getdata())
    values = array("f")
    for channel in range(3):
        values.extend((pixel[channel] / 255.0 - 0.5) / 0.5 for pixel in pixels)
    if sys.byteorder != "little":
        values.byteswap()
    with raw_path.open("wb") as f:
        values.tofile(f)


def deterministic_embedding(payload: bytes, dim: int = 768) -> list[float]:
    """Create a deterministic normalized vector for local wiring tests."""
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(payload + counter.to_bytes(4, "little")).digest()
        counter += 1
        for idx in range(0, len(digest), 4):
            if len(values) >= dim:
                break
            integer = int.from_bytes(digest[idx : idx + 4], "little", signed=False)
            values.append((integer / 2**32) * 2.0 - 1.0)
    return l2_normalize(values, expected_dim=dim)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")

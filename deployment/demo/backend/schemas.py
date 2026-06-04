"""Pydantic schemas for the local FastAPI demo backend."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CollapseKey = Literal["episode_id", "track_id", "id", "none"]


class EmbeddingRecordPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    board_id: str
    camera_id: str
    track_id: str
    episode_id: str
    captured_at: str
    bbox: tuple[int, int, int, int]
    quality_score: float
    embedding: list[float] = Field(min_length=768, max_length=768)
    crop_path: str
    model_version: str
    save_reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("embedding contains NaN or Inf")
        return values


class TrackEmbeddingResponse(BaseModel):
    status: Literal["ok"]
    embedding_id: str


class SearchTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str
    camera_ids: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    top_k: int = Field(default=10, ge=1)
    top_k_raw: int = Field(default=50, ge=1)
    collapse_key: CollapseKey = "episode_id"
    include_embeddings: bool = False


class SearchResultPayload(BaseModel):
    score: float
    embedding_id: str
    episode_id: str
    track_id: str
    snapshot_count: int
    camera_id: str
    board_id: str
    captured_at: str
    first_seen: str | None = None
    last_seen: str | None = None
    bbox: tuple[int, int, int, int]
    quality_score: float
    crop_path: str
    model_version: str
    save_reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class SearchTextResponse(BaseModel):
    query_text: str
    query_id: str
    text_encoder: str
    top_k_raw: int
    top_k_final: int
    collapse_key: CollapseKey
    results: list[SearchResultPayload]


class HeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board_id: str
    camera_id: str
    encoder_runtime: str
    upload_backlog: int = Field(ge=0)
    spool_pending: int = Field(default=0, ge=0)
    spool_sent: int = Field(default=0, ge=0)
    spool_failed: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HeartbeatResponse(BaseModel):
    status: Literal["ok"]
    heartbeat_id: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    db_path: str
    record_count: int
    heartbeat_count: int
    latest_heartbeat: dict[str, Any] | None = None

"""FastAPI app for the local modular demo backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from ..adapters.encoders import FakeTextEncoder
from ..core.contracts import EmbeddingRecord
from ..core.utils import stable_id, utc_now_iso
from .schemas import (
    EmbeddingRecordPayload,
    HealthResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    SearchResultPayload,
    SearchTextRequest,
    SearchTextResponse,
    TrackEmbeddingResponse,
)
from .storage import BackendSearchResult, LocalBackendStore


DEFAULT_DB_PATH = Path("artifacts/deployment/runtime/backend/demo_backend.sqlite3")


def _record_from_payload(payload: EmbeddingRecordPayload) -> EmbeddingRecord:
    data = payload.model_dump()
    return EmbeddingRecord(**data)


def _search_result_payload(
    result: BackendSearchResult,
    include_embedding: bool,
) -> dict[str, Any]:
    record = result.record
    first_seen = record.metadata.get("first_seen")
    last_seen = record.metadata.get("last_seen")
    payload = SearchResultPayload(
        score=result.score,
        embedding_id=record.id,
        episode_id=record.episode_id,
        track_id=record.track_id,
        snapshot_count=result.snapshot_count,
        camera_id=record.camera_id,
        board_id=record.board_id,
        captured_at=record.captured_at,
        first_seen=first_seen,
        last_seen=last_seen,
        bbox=record.bbox,
        quality_score=record.quality_score,
        crop_path=record.crop_path,
        model_version=record.model_version,
        save_reason=record.save_reason,
        metadata=record.metadata,
        embedding=record.embedding if include_embedding else None,
    )
    data = payload.model_dump()
    if not include_embedding:
        data.pop("embedding", None)
    return data


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> FastAPI:
    store = LocalBackendStore(Path(db_path))
    text_encoder = FakeTextEncoder()
    app = FastAPI(
        title="mSigLIP Modular Demo Backend",
        version="0.1.0",
        description="Local FastAPI shell for edge ingest, heartbeat, and text search.",
    )
    app.state.demo_store = store

    @app.post(
        "/api/v1/ingest/track-embedding",
        response_model=TrackEmbeddingResponse,
    )
    def ingest_track_embedding(payload: EmbeddingRecordPayload) -> TrackEmbeddingResponse:
        try:
            embedding_id = store.upsert_embedding(_record_from_payload(payload))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return TrackEmbeddingResponse(status="ok", embedding_id=embedding_id)

    @app.post(
        "/api/v1/search/text",
        response_model=SearchTextResponse,
        response_model_exclude_none=True,
    )
    def search_text(payload: SearchTextRequest) -> SearchTextResponse:
        try:
            query_embedding = text_encoder.encode(payload.query_text)
            results = store.search(
                query_embedding=query_embedding,
                top_k_raw=payload.top_k_raw,
                top_k_final=payload.top_k,
                camera_ids=payload.camera_ids,
                start_time=payload.start_time,
                end_time=payload.end_time,
                collapse_key=payload.collapse_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        query_id = stable_id(payload.query_text, utc_now_iso())
        return SearchTextResponse(
            query_text=payload.query_text,
            query_id=query_id,
            text_encoder=text_encoder.runtime_name,
            top_k_raw=payload.top_k_raw,
            top_k_final=payload.top_k,
            collapse_key=payload.collapse_key,
            results=[
                SearchResultPayload(**_search_result_payload(result, payload.include_embeddings))
                for result in results
            ],
        )

    @app.post("/api/v1/boards/heartbeat", response_model=HeartbeatResponse)
    def boards_heartbeat(payload: HeartbeatRequest) -> HeartbeatResponse:
        heartbeat_id = store.add_heartbeat(payload.model_dump())
        return HeartbeatResponse(status="ok", heartbeat_id=heartbeat_id)

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            db_path=str(store.db_path),
            record_count=store.record_count(),
            heartbeat_count=store.heartbeat_count(),
            latest_heartbeat=store.latest_heartbeat(),
        )

    return app

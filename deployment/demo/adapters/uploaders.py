"""Uploader adapters."""

from __future__ import annotations

from typing import Any

import httpx

from ..core.contracts import EmbeddingRecord, UploadResult
from .vector_store import JsonlVectorStore


class LocalVectorStoreUploader:
    """Commit upload events into the local JSONL vector store."""

    def __init__(self, vector_store: JsonlVectorStore):
        self.vector_store = vector_store

    def upload(self, event: dict) -> UploadResult:
        record_data = event.get("record")
        if not record_data:
            return UploadResult(ok=False, error="event missing record")
        record_data = dict(record_data)
        record_data["bbox"] = tuple(record_data["bbox"])
        record = EmbeddingRecord(**record_data)
        self.vector_store.upsert(record)
        return UploadResult(ok=True, remote_id=record.id)


class FailingUploader:
    """Uploader used to verify failed spool behavior."""

    def __init__(self, error: str = "intentional upload failure"):
        self.error = error

    def upload(self, event: dict) -> UploadResult:
        return UploadResult(ok=False, error=self.error)


class HttpUploader:
    """Upload embedding events to the FastAPI demo backend."""

    def __init__(
        self,
        backend_url: str,
        board_token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.board_token = board_token
        self.timeout = timeout
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.board_token:
            headers["Authorization"] = f"Bearer {self.board_token}"
        return headers

    def _record_payload(self, event: dict[str, Any]) -> dict[str, Any] | None:
        record_data = event.get("record")
        if not isinstance(record_data, dict):
            return None
        return record_data

    def upload(self, event: dict) -> UploadResult:
        record_data = self._record_payload(event)
        if record_data is None:
            return UploadResult(ok=False, error="event missing record")
        url = f"{self.backend_url}/api/v1/ingest/track-embedding"
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(url, json=record_data, headers=self._headers())
        except httpx.HTTPError as exc:
            return UploadResult(ok=False, error=str(exc))

        if response.status_code < 200 or response.status_code >= 300:
            return UploadResult(
                ok=False,
                error=f"HTTP {response.status_code}: {response.text}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            return UploadResult(ok=False, error=f"invalid JSON response: {exc}")
        if payload.get("status") != "ok":
            return UploadResult(ok=False, error=f"backend rejected upload: {payload}")
        return UploadResult(ok=True, remote_id=payload.get("embedding_id"))

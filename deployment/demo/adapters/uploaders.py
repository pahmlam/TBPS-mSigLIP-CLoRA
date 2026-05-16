"""Uploader adapters."""

from __future__ import annotations

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

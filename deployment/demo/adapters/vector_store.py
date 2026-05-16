"""Local JSONL vector store for demo search."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..core.contracts import EmbeddingRecord, SearchResult
from ..core.utils import cosine, ensure_dir, l2_normalize


def _record_from_dict(data: dict) -> EmbeddingRecord:
    data = dict(data)
    data["bbox"] = tuple(data["bbox"])
    data["embedding"] = [float(value) for value in data["embedding"]]
    return EmbeddingRecord(**data)


class JsonlVectorStore:
    """Dependency-light vector store for local and RB3 system wiring tests."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        ensure_dir(self.path.parent)

    def upsert(self, record: EmbeddingRecord) -> None:
        normalized = l2_normalize([float(value) for value in record.embedding])
        record = EmbeddingRecord(**{**asdict(record), "embedding": normalized})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _records(self) -> list[EmbeddingRecord]:
        if not self.path.exists():
            return []
        records: list[EmbeddingRecord] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(_record_from_dict(json.loads(line)))
        return records

    @staticmethod
    def _passes_filters(
        record: EmbeddingRecord,
        camera_ids: list[str] | None,
        start_time: str | None,
        end_time: str | None,
    ) -> bool:
        if camera_ids and record.camera_id not in camera_ids:
            return False
        if start_time and record.captured_at < start_time:
            return False
        if end_time and record.captured_at > end_time:
            return False
        return True

    @staticmethod
    def _collapse(results: list[SearchResult], collapse_key: str) -> list[SearchResult]:
        if collapse_key not in {"episode_id", "track_id", "id", "none"}:
            raise ValueError(f"Unsupported collapse_key: {collapse_key}")
        if collapse_key == "none":
            return results
        best: dict[str, SearchResult] = {}
        for result in results:
            key = getattr(result.record, collapse_key)
            existing = best.get(key)
            if existing is None or result.score > existing.score:
                best[key] = result
        return sorted(best.values(), key=lambda item: item.score, reverse=True)

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
        query = l2_normalize([float(value) for value in query_embedding])
        scored = [
            SearchResult(record=record, score=cosine(query, record.embedding))
            for record in self._records()
            if self._passes_filters(record, camera_ids, start_time, end_time)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        collapsed = self._collapse(scored[:top_k_raw], collapse_key=collapse_key)
        return collapsed[:top_k_final]

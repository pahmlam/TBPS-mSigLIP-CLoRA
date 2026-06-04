"""SQLite-backed local storage for the FastAPI demo backend."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.contracts import EmbeddingRecord
from ..core.utils import cosine, ensure_dir, l2_normalize, utc_now_iso


@dataclass(frozen=True)
class BackendSearchResult:
    record: EmbeddingRecord
    score: float
    snapshot_count: int


def _record_from_row(row: sqlite3.Row) -> EmbeddingRecord:
    metadata = json.loads(row["metadata_json"])
    embedding = [float(value) for value in json.loads(row["embedding_json"])]
    bbox = tuple(json.loads(row["bbox_json"]))
    return EmbeddingRecord(
        id=row["id"],
        board_id=row["board_id"],
        camera_id=row["camera_id"],
        track_id=row["track_id"],
        episode_id=row["episode_id"],
        captured_at=row["captured_at"],
        bbox=bbox,
        quality_score=float(row["quality_score"]),
        embedding=embedding,
        crop_path=row["crop_path"],
        model_version=row["model_version"],
        save_reason=row["save_reason"],
        metadata=metadata,
    )


class LocalBackendStore:
    """Small SQLite store used to lock the demo API contract before Supabase."""

    def __init__(self, db_path: Path):
        self.db_path = db_path.expanduser().resolve()
        ensure_dir(self.db_path.parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                create table if not exists embeddings (
                  id text primary key,
                  board_id text not null,
                  camera_id text not null,
                  track_id text not null,
                  episode_id text not null,
                  captured_at text not null,
                  bbox_json text not null,
                  quality_score real not null,
                  embedding_json text not null,
                  crop_path text not null,
                  model_version text not null,
                  save_reason text not null,
                  metadata_json text not null,
                  created_at text not null
                )
                """
            )
            con.execute(
                """
                create index if not exists idx_embeddings_camera_time
                on embeddings (camera_id, captured_at)
                """
            )
            con.execute(
                """
                create index if not exists idx_embeddings_episode
                on embeddings (episode_id)
                """
            )
            con.execute(
                """
                create table if not exists heartbeats (
                  id integer primary key autoincrement,
                  board_id text not null,
                  camera_id text not null,
                  encoder_runtime text not null,
                  upload_backlog integer not null,
                  spool_pending integer not null,
                  spool_sent integer not null,
                  spool_failed integer not null,
                  metadata_json text not null,
                  created_at text not null
                )
                """
            )

    def upsert_embedding(self, record: EmbeddingRecord) -> str:
        normalized = l2_normalize([float(value) for value in record.embedding])
        with self._connect() as con:
            con.execute(
                """
                insert into embeddings (
                  id, board_id, camera_id, track_id, episode_id, captured_at,
                  bbox_json, quality_score, embedding_json, crop_path,
                  model_version, save_reason, metadata_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  board_id=excluded.board_id,
                  camera_id=excluded.camera_id,
                  track_id=excluded.track_id,
                  episode_id=excluded.episode_id,
                  captured_at=excluded.captured_at,
                  bbox_json=excluded.bbox_json,
                  quality_score=excluded.quality_score,
                  embedding_json=excluded.embedding_json,
                  crop_path=excluded.crop_path,
                  model_version=excluded.model_version,
                  save_reason=excluded.save_reason,
                  metadata_json=excluded.metadata_json
                """,
                (
                    record.id,
                    record.board_id,
                    record.camera_id,
                    record.track_id,
                    record.episode_id,
                    record.captured_at,
                    json.dumps(list(record.bbox)),
                    float(record.quality_score),
                    json.dumps(normalized),
                    record.crop_path,
                    record.model_version,
                    record.save_reason,
                    json.dumps(record.metadata, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
        return record.id

    def _records(
        self,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[EmbeddingRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if camera_ids:
            placeholders = ", ".join("?" for _ in camera_ids)
            clauses.append(f"camera_id in ({placeholders})")
            params.extend(camera_ids)
        if start_time:
            clauses.append("captured_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("captured_at <= ?")
            params.append(end_time)
        where = f"where {' and '.join(clauses)}" if clauses else ""
        with self._connect() as con:
            rows = con.execute(f"select * from embeddings {where}", params).fetchall()
        return [_record_from_row(row) for row in rows]

    @staticmethod
    def _collapse(
        results: list[BackendSearchResult],
        collapse_key: str,
    ) -> list[BackendSearchResult]:
        if collapse_key not in {"episode_id", "track_id", "id", "none"}:
            raise ValueError(f"Unsupported collapse_key: {collapse_key}")
        if collapse_key == "none":
            return results

        groups: dict[str, BackendSearchResult] = {}
        counts: dict[str, int] = {}
        for result in results:
            key = getattr(result.record, collapse_key)
            counts[key] = counts.get(key, 0) + 1
            current = groups.get(key)
            if current is None or result.score > current.score:
                groups[key] = result

        collapsed = [
            BackendSearchResult(
                record=result.record,
                score=result.score,
                snapshot_count=counts[getattr(result.record, collapse_key)],
            )
            for result in groups.values()
        ]
        return sorted(collapsed, key=lambda item: item.score, reverse=True)

    def search(
        self,
        query_embedding: list[float],
        top_k_raw: int,
        top_k_final: int,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        collapse_key: str = "episode_id",
    ) -> list[BackendSearchResult]:
        query = l2_normalize([float(value) for value in query_embedding])
        scored = [
            BackendSearchResult(
                record=record,
                score=cosine(query, record.embedding),
                snapshot_count=1,
            )
            for record in self._records(camera_ids, start_time, end_time)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        collapsed = self._collapse(scored[:top_k_raw], collapse_key=collapse_key)
        return collapsed[:top_k_final]

    def record_count(self) -> int:
        with self._connect() as con:
            row = con.execute("select count(*) as count from embeddings").fetchone()
        return int(row["count"])

    def add_heartbeat(self, payload: dict[str, Any]) -> int:
        with self._connect() as con:
            cur = con.execute(
                """
                insert into heartbeats (
                  board_id, camera_id, encoder_runtime, upload_backlog,
                  spool_pending, spool_sent, spool_failed, metadata_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["board_id"],
                    payload["camera_id"],
                    payload["encoder_runtime"],
                    int(payload["upload_backlog"]),
                    int(payload.get("spool_pending", 0)),
                    int(payload.get("spool_sent", 0)),
                    int(payload.get("spool_failed", 0)),
                    json.dumps(payload.get("metadata", {}), ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def heartbeat_count(self) -> int:
        with self._connect() as con:
            row = con.execute("select count(*) as count from heartbeats").fetchone()
        return int(row["count"])

    def latest_heartbeat(self) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                """
                select * from heartbeats
                order by id desc
                limit 1
                """
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload

    def dump_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records()]

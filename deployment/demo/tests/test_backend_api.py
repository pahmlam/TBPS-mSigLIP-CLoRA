from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from fastapi.testclient import TestClient

from deployment.demo.adapters.encoders import FakeTextEncoder
from deployment.demo.backend.app import create_app
from deployment.demo.core.contracts import EmbeddingRecord
from deployment.demo.core.utils import deterministic_embedding


def _record(record_id: str, episode_id: str, embedding: list[float], camera_id: str = "cam-a") -> dict:
    return asdict(
        EmbeddingRecord(
            id=record_id,
            board_id="board-test",
            camera_id=camera_id,
            track_id=f"trk-{record_id}",
            episode_id=episode_id,
            captured_at="2026-06-04T10:00:00Z",
            bbox=(0, 0, 64, 128),
            quality_score=0.9,
            embedding=embedding,
            crop_path=f"/tmp/{record_id}.jpg",
            model_version="test-model",
            save_reason="best_frame",
            metadata={"first_seen": "2026-06-04T09:59:58Z", "last_seen": "2026-06-04T10:00:01Z"},
        )
    )


class BackendApiTest(unittest.TestCase):
    def test_ingest_normalizes_vector_and_health_counts_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            app = create_app(Path(raw_tmp) / "backend.sqlite3")
            client = TestClient(app)

            payload = _record("rec-1", "episode-1", [1.0] * 768)
            response = client.post("/api/v1/ingest/track-embedding", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["embedding_id"], "rec-1")

            records = app.state.demo_store._records()
            self.assertEqual(len(records), 1)
            norm = sum(value * value for value in records[0].embedding) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=6)

            health = client.get("/api/v1/health").json()
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["record_count"], 1)

    def test_search_filters_and_collapses_by_episode(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            app = create_app(Path(raw_tmp) / "backend.sqlite3")
            client = TestClient(app)
            query_text = "người mặc áo đỏ"
            query_embedding = FakeTextEncoder().encode(query_text)

            records = [
                _record("same-low", "episode-1", deterministic_embedding(b"far"), camera_id="cam-a"),
                _record("same-high", "episode-1", query_embedding, camera_id="cam-a"),
                _record("other-camera", "episode-2", query_embedding, camera_id="cam-b"),
            ]
            for record in records:
                response = client.post("/api/v1/ingest/track-embedding", json=record)
                self.assertEqual(response.status_code, 200)

            response = client.post(
                "/api/v1/search/text",
                json={
                    "query_text": query_text,
                    "camera_ids": ["cam-a"],
                    "top_k": 10,
                    "top_k_raw": 10,
                    "collapse_key": "episode_id",
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()

            self.assertEqual(payload["text_encoder"], "fake-text")
            self.assertEqual(len(payload["results"]), 1)
            result = payload["results"][0]
            self.assertEqual(result["embedding_id"], "same-high")
            self.assertEqual(result["episode_id"], "episode-1")
            self.assertEqual(result["snapshot_count"], 2)
            self.assertNotIn("embedding", result)

    def test_heartbeat_updates_health(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            app = create_app(Path(raw_tmp) / "backend.sqlite3")
            client = TestClient(app)

            response = client.post(
                "/api/v1/boards/heartbeat",
                json={
                    "board_id": "qc-rb3g2",
                    "camera_id": "cam-lab-01",
                    "encoder_runtime": "fake-vision",
                    "upload_backlog": 2,
                    "spool_pending": 1,
                    "spool_sent": 3,
                    "spool_failed": 1,
                    "metadata": {"source": "unit-test"},
                },
            )
            self.assertEqual(response.status_code, 200)

            health = client.get("/api/v1/health").json()
            self.assertEqual(health["heartbeat_count"], 1)
            self.assertEqual(health["latest_heartbeat"]["board_id"], "qc-rb3g2")
            self.assertEqual(health["latest_heartbeat"]["metadata"]["source"], "unit-test")


if __name__ == "__main__":
    unittest.main()

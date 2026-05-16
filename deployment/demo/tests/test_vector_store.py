from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deployment.demo.adapters.vector_store import JsonlVectorStore
from deployment.demo.core.contracts import EmbeddingRecord
from deployment.demo.core.utils import deterministic_embedding


def _record(record_id: str, episode_id: str, payload: bytes) -> EmbeddingRecord:
    return EmbeddingRecord(
        id=record_id,
        board_id="board-test",
        camera_id="cam-test",
        track_id=f"trk-{record_id}",
        episode_id=episode_id,
        captured_at="2026-05-16T00:00:00Z",
        bbox=(0, 0, 64, 128),
        quality_score=1.0,
        embedding=deterministic_embedding(payload),
        crop_path=f"/tmp/{record_id}.jpg",
        model_version="test",
        save_reason="best_frame",
    )


class JsonlVectorStoreTest(unittest.TestCase):
    def test_search_collapses_by_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlVectorStore(Path(tmp) / "vectors.jsonl")
            query = deterministic_embedding(b"query")
            same_episode_low = _record("a", "episode-1", b"far")
            same_episode_high = EmbeddingRecord(
                **{**same_episode_low.__dict__, "id": "b", "embedding": query}
            )
            other_episode = _record("c", "episode-2", b"other")
            store.upsert(same_episode_low)
            store.upsert(same_episode_high)
            store.upsert(other_episode)

            results = store.search(query, top_k_raw=10, top_k_final=10, collapse_key="episode_id")

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].record.id, "b")
            self.assertEqual(results[0].record.episode_id, "episode-1")
            self.assertAlmostEqual(results[0].score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

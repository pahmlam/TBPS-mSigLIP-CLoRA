from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from deployment.demo.adapters.crop_selector import DefaultCropSelector
from deployment.demo.adapters.detectors import FullFramePersonDetector
from deployment.demo.adapters.encoders import FakeVisionEncoder
from deployment.demo.adapters.sources import ImageDirectorySource
from deployment.demo.adapters.spool import DiskSpool
from deployment.demo.adapters.tracker import SimpleTracker
from deployment.demo.adapters.uploaders import FailingUploader, LocalVectorStoreUploader
from deployment.demo.adapters.vector_store import JsonlVectorStore
from deployment.demo.core.pipeline import IngestPipeline


def _write_image(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (64, 128), color=(120, 30, 200))
    image.save(path)


class IngestPipelineTest(unittest.TestCase):
    def _pipeline(self, tmp: Path, upload_mode: str = "local") -> tuple[IngestPipeline, JsonlVectorStore, DiskSpool]:
        image_dir = tmp / "images"
        image_dir.mkdir()
        _write_image(image_dir / "person.jpg")
        store = JsonlVectorStore(tmp / "vectors.jsonl")
        spool = DiskSpool(tmp / "spool")
        uploader = LocalVectorStoreUploader(store) if upload_mode == "local" else FailingUploader()
        pipeline = IngestPipeline(
            source=ImageDirectorySource(image_dir),
            detector=FullFramePersonDetector(),
            tracker=SimpleTracker(mode="per_frame"),
            crop_selector=DefaultCropSelector(),
            image_encoder=FakeVisionEncoder(),
            spool=spool,
            uploader=uploader,
            crop_dir=tmp / "crops",
            board_id="board-test",
            camera_id="cam-test",
            model_version="test-model",
        )
        return pipeline, store, spool

    def test_ingest_writes_vector_and_sent_spool(self) -> None:
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow is not installed in this Python environment")
        with tempfile.TemporaryDirectory() as raw_tmp:
            pipeline, store, spool = self._pipeline(Path(raw_tmp), upload_mode="local")
            stats = pipeline.run()

            self.assertEqual(stats.frames_seen, 1)
            self.assertEqual(stats.embeddings_written, 1)
            self.assertEqual(spool.counts(), {"pending": 0, "sent": 1, "failed": 0})
            self.assertEqual(len(store._records()), 1)
            self.assertEqual(len(list((Path(raw_tmp) / "crops").glob("*.jpg"))), 1)

    def test_failed_upload_moves_event_to_failed_spool(self) -> None:
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow is not installed in this Python environment")
        with tempfile.TemporaryDirectory() as raw_tmp:
            pipeline, store, spool = self._pipeline(Path(raw_tmp), upload_mode="fail")
            stats = pipeline.run()

            self.assertEqual(stats.uploads_failed, 1)
            self.assertEqual(spool.counts(), {"pending": 0, "sent": 0, "failed": 1})
            self.assertEqual(len(store._records()), 0)


if __name__ == "__main__":
    unittest.main()

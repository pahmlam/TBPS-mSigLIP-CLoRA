from __future__ import annotations

import unittest

import httpx

from deployment.demo.adapters.uploaders import HttpUploader


class HttpUploaderTest(unittest.TestCase):
    def test_upload_maps_success_response(self) -> None:
        seen_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers["authorization"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={"status": "ok", "embedding_id": "rec-1"})

        uploader = HttpUploader(
            backend_url="http://demo-backend",
            board_token="board-secret",
            transport=httpx.MockTransport(handler),
        )
        result = uploader.upload({"record": {"id": "rec-1"}})

        self.assertTrue(result.ok)
        self.assertEqual(result.remote_id, "rec-1")
        self.assertEqual(seen_headers["authorization"], "Bearer board-secret")

    def test_upload_maps_http_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        uploader = HttpUploader(
            backend_url="http://demo-backend",
            transport=httpx.MockTransport(handler),
        )
        result = uploader.upload({"record": {"id": "rec-1"}})

        self.assertFalse(result.ok)
        self.assertIn("HTTP 500", result.error)


if __name__ == "__main__":
    unittest.main()

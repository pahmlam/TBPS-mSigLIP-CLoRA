"""Disk-backed spool for upload/retry behavior."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..core.contracts import UploadResult
from ..core.utils import ensure_dir, stable_id, to_jsonable, utc_now_iso, write_json


class DiskSpool:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.pending_dir = ensure_dir(self.root / "pending")
        self.sent_dir = ensure_dir(self.root / "sent")
        self.failed_dir = ensure_dir(self.root / "failed")

    def enqueue(self, event: dict[str, Any]) -> str:
        spool_id = event.get("spool_id") or stable_id(event.get("record", {}), utc_now_iso())
        payload = {
            "spool_id": spool_id,
            "created_at": utc_now_iso(),
            "retry_count": int(event.get("retry_count", 0)),
            "event": to_jsonable(event),
        }
        write_json(self.pending_dir / f"{spool_id}.json", payload)
        return spool_id

    def _move_with_result(self, spool_id: str, target_dir: Path, result: UploadResult) -> None:
        src = self.pending_dir / f"{spool_id}.json"
        if not src.exists():
            src = self.failed_dir / f"{spool_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"Spool event not found: {spool_id}")
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload["completed_at"] = utc_now_iso()
        payload["upload_result"] = to_jsonable(result)
        dst = target_dir / f"{spool_id}.json"
        write_json(dst, payload)
        if src != dst:
            src.unlink()

    def mark_sent(self, spool_id: str, result: UploadResult) -> None:
        self._move_with_result(spool_id, self.sent_dir, result)

    def mark_failed(self, spool_id: str, result: UploadResult) -> None:
        self._move_with_result(spool_id, self.failed_dir, result)

    def retry_failed(self, spool_id: str) -> None:
        src = self.failed_dir / f"{spool_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"Failed spool event not found: {spool_id}")
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload["retry_count"] = int(payload.get("retry_count", 0)) + 1
        dst = self.pending_dir / src.name
        write_json(dst, payload)
        src.unlink()

    def iter_pending(self) -> Iterable[dict[str, Any]]:
        for path in sorted(self.pending_dir.glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))

    def counts(self) -> dict[str, int]:
        return {
            "pending": len(list(self.pending_dir.glob("*.json"))),
            "sent": len(list(self.sent_dir.glob("*.json"))),
            "failed": len(list(self.failed_dir.glob("*.json"))),
        }

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.pending_dir = ensure_dir(self.root / "pending")
        self.sent_dir = ensure_dir(self.root / "sent")
        self.failed_dir = ensure_dir(self.root / "failed")

"""CLI for checking local demo runtime health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..adapters.spool import DiskSpool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print modular demo runtime health.")
    parser.add_argument("--board-id", default="local-dev")
    parser.add_argument("--camera-id", default="cam-local-01")
    parser.add_argument("--encoder-runtime", default="unknown")
    parser.add_argument("--store", default="deployment/demo_runtime/vectors.jsonl")
    parser.add_argument("--spool", default="deployment/demo_runtime/spool")
    return parser.parse_args()


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    spool = DiskSpool(Path(args.spool))
    counts = spool.counts()
    payload = {
        "board_id": args.board_id,
        "camera_id": args.camera_id,
        "encoder_runtime": args.encoder_runtime,
        "vector_records": _line_count(Path(args.store)),
        "upload_backlog": counts["pending"] + counts["failed"],
        "spool": counts,
        "rb3_acceptance_required": True,
        "note": "Local health is a preflight check only; deployment acceptance must run on RB3 with QNN.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

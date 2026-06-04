"""CLI entrypoint for the local FastAPI demo backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import DEFAULT_DB_PATH, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mSigLIP demo FastAPI backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.db))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

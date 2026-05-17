"""CLI for local vector search with result collapse."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .builders import build_text_encoder
from ..adapters.vector_store import JsonlVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the local demo vector store.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--store", default="artifacts/deployment/runtime/vectors.jsonl")
    parser.add_argument("--text-encoder", choices=["fake"], default="fake")
    parser.add_argument("--top-k-raw", type=int, default=50)
    parser.add_argument("--top-k-final", type=int, default=10)
    parser.add_argument("--collapse-key", choices=["episode_id", "track_id", "id", "none"], default="episode_id")
    parser.add_argument("--camera-id", action="append", dest="camera_ids")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--include-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text_encoder = build_text_encoder(args)
    store = JsonlVectorStore(Path(args.store))
    query_embedding = text_encoder.encode(args.query)
    results = store.search(
        query_embedding=query_embedding,
        top_k_raw=args.top_k_raw,
        top_k_final=args.top_k_final,
        camera_ids=args.camera_ids,
        start_time=args.start_time,
        end_time=args.end_time,
        collapse_key=args.collapse_key,
    )
    output_results = []
    for result in results:
        record = asdict(result.record)
        if not args.include_embeddings:
            record.pop("embedding", None)
        output_results.append({"score": result.score, "record": record})

    payload = {
        "query": args.query,
        "text_encoder": text_encoder.runtime_name,
        "top_k_raw": args.top_k_raw,
        "top_k_final": args.top_k_final,
        "collapse_key": args.collapse_key,
        "results": output_results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

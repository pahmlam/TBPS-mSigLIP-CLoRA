#!/usr/bin/env python3
"""List recent Qualcomm AI Hub datasets and their IDs."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List recent QAI Hub datasets.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import qai_hub as hub

    datasets = hub.get_datasets(offset=args.offset, limit=args.limit)
    if not datasets:
        print("No datasets found.")
        return

    for dataset in datasets:
        dataset_id = getattr(dataset, "dataset_id", "")
        name = (
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "name", None)
            or getattr(dataset, "dataset_display_name", None)
            or "<unknown>"
        )
        created = getattr(dataset, "creation_time", "")
        expires = getattr(dataset, "expiration_time", "")
        print(f"{dataset_id}\t{name}\tcreated={created}\texpires={expires}")


if __name__ == "__main__":
    main()

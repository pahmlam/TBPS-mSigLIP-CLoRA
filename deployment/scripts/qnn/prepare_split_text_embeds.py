#!/usr/bin/env python3
"""Host prep: VN3K captions -> split-text `inputs_embeds` (calib / query / zero control).

The deployable text path is split-encoder (host embedding lookup + HTP transformer,
`w8a8_qat_rotated.md` §12A.8). Several input sets are `inputs_embeds` produced by the
host CPU step: the AI Hub calibration set (train captions), the full-test query set
(every test caption, for the gate-comparable retrieval), and a zero-embeds control.

This script is the reproducible generator for all of them. It tokenizes VN3K captions
with the project tokenizer and looks them up in the dumped token-embedding table
(`dump_text_embedding_table.py`), reusing `board_text_encode.py` for both. It writes
`inputs_embeds` + `attention_mask` raw files, an `input_list.txt`, and a `manifest.csv`
(pid=`id-1`, matching `prepare_vn3k_text_inputs.py`), ready for AI Hub calibration upload
or `qnn-net-run` with `text_encoder_split.bin`.

Examples:
    # AI Hub calibration source (train, 500 captions)
    prepare_split_text_embeds.py --split train --num-samples 500 --output-dir .../calib_500_split_embeds
    # full-test query set (every test caption, 4000) — the gate-comparable retrieval set
    prepare_split_text_embeds.py --split test --all-captions --output-dir .../query_full_split_embeds
    # zero-embeds control (same masks, zeroed embeds) — proves the board graph uses embeds
    prepare_split_text_embeds.py --split test --all-captions --zero-embeds --output-dir .../query_full_zero
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from board_text_encode import _build_lookup, _tokenize_captions  # noqa: E402


def _resolve_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "data_captions_vn3k.json").exists():
        return path
    if (path / "VN3K" / "data_captions_vn3k.json").exists():
        return path / "VN3K"
    raise SystemExit(f"data_captions_vn3k.json not found under {path}")


def _load_captions(root: Path, split: str, all_captions: bool) -> list[tuple[int, str, str]]:
    annos = json.loads((root / "data_captions_vn3k.json").read_text(encoding="utf-8"))
    recs: list[tuple[int, str, str]] = []
    for a in annos:
        if a.get("split", "") != split:
            continue
        raw = a.get("captions")
        caps = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if not isinstance(caps, list):
            caps = [str(caps)]
        caps = [str(c).strip() for c in caps if str(c).strip()]
        if not caps:
            continue
        pid, fp = int(a["id"]) - 1, a.get("file_path", "")
        for c in (caps if all_captions else caps[:1]):
            recs.append((pid, c, fp))
    if not recs:
        raise SystemExit(f"No captions for split={split!r}")
    return recs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VN3K captions -> split-text inputs_embeds.")
    p.add_argument("--vn3k-root", type=Path, default=PROJECT_ROOT / "VN3K")
    p.add_argument("--split", choices=["train", "test", "validate"], default="test")
    p.add_argument("--all-captions", action="store_true", help="Every caption per image (else first only).")
    p.add_argument("--num-samples", type=int, default=0, help="0 = all selected; else cap to N.")
    p.add_argument("--selection", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=2400)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--zero-embeds", action="store_true", help="Write zeroed inputs_embeds (control); keep masks.")
    p.add_argument("--table", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/bin/token_embedding_v8/token_embedding_int8.bin")
    p.add_argument("--meta", type=Path)
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8",
                   help="Model dir with config.yaml for the tokenizer.")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve_root(args.vn3k_root)
    recs = _load_captions(root, args.split, args.all_captions)
    if args.selection == "random":
        random.Random(args.seed).shuffle(recs)
    recs = recs[args.start_index:]
    if args.num_samples:
        recs = recs[: args.num_samples]

    lookup, vocab, dim = _build_lookup(args.table.expanduser().resolve(),
                                       args.meta.expanduser().resolve() if args.meta else None)
    captions = [c for _, c, _ in recs]
    ids_list, mask_list = _tokenize_captions(captions, args.model_dir.expanduser().resolve(), args.seq_len)

    out = args.output_dir.expanduser().resolve()
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lines, manifest = [], []
    for i, ((pid, cap, _), ids, mask) in enumerate(zip(recs, ids_list, mask_list)):
        ids = np.clip(ids, 0, vocab - 1)
        embeds = lookup(ids).astype(np.float32).reshape(1, args.seq_len, dim)
        if args.zero_embeds:
            embeds = np.zeros_like(embeds)
        stem = f"{i:05d}_pid{pid}"
        ep, mp = raw / f"{stem}_inputs_embeds.raw", raw / f"{stem}_attention_mask.raw"
        embeds.tofile(ep)
        mask.astype(np.float32).tofile(mp)
        lines.append(f"inputs_embeds:=raw/{ep.name} attention_mask:=raw/{mp.name}")
        manifest.append({"sample_index": i, "pid": pid, "caption": cap[:120], "num_tokens": int(mask.sum())})

    (out / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_index", "pid", "caption", "num_tokens"])
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote {len(recs)} inputs_embeds{' (ZEROED)' if args.zero_embeds else ''} -> {out}")


if __name__ == "__main__":
    main()

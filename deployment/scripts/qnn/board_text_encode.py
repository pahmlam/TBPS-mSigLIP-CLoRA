#!/usr/bin/env python3
"""On-device (RB3 ARM CPU) half of the split text encoder: tokenize + embedding lookup.

The split text encoder runs the transformer on the HTP NPU (`text_encoder_split.bin`),
but the embedding lookup `inputs_embeds = token_embedding[input_ids]` must happen on a
CPU because HTP v68 does not honour the dynamic 250k-row gather (see
`w8a8_qat_rotated.md` §12A.8). This script is that CPU step, designed to run **on the
RB3 itself** so the whole text encoding pipeline is on-device:

    caption --(tokenizer, optional)--> input_ids
            --(this script: table lookup)--> inputs_embeds .raw  --> qnn-net-run(split bin)

It is intentionally light: the lookup needs only numpy (the table is memory-mapped, so
it does not load all 384 MB into RAM). Tokenization is optional — if the project
tokenizer + transformers are available on the board, pass `--captions`/`--model-dir`;
otherwise feed a directory of pre-tokenized `input_ids` raw files (`--input-dir`).

Table: dump once on the host with the rotated v8 model
(`token_embedding_v8/token_embedding_fp16.bin` + `meta.json`). It is a frozen model
weight (rotated `token_embedding.weight`); regenerate only if the text model changes.

Output: `inputs_embeds` + `attention_mask` raw files, an `input_list.txt`, and a
`manifest.csv` (pid/caption), ready for `qnn-net-run` with `text_encoder_split.bin`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _find_meta(table_path: Path, meta_arg: Path | None) -> dict:
    if meta_arg is not None:
        return json.loads(meta_arg.read_text())
    metas = []
    for mp in sorted(table_path.parent.glob("meta*.json")):
        try:
            metas.append(json.loads(mp.read_text()))
        except Exception:
            continue
    # 1) exact match: a meta that names this table as its weight file (int8 per-row tables).
    for meta in metas:
        if meta.get("weight_file") == table_path.name:
            return meta
    # 2) float tables: meta describing a float dtype with no explicit weight_file.
    for meta in metas:
        if "weight_file" not in meta and meta.get("dtype") in ("float16", "float32"):
            return meta
    fallback = table_path.parent / "meta.json"
    if fallback.exists():
        return json.loads(fallback.read_text())
    raise SystemExit(f"No meta json found next to table: {table_path.parent}")


def _build_lookup(table_path: Path, meta_arg: Path | None):
    """Return (lookup_fn, vocab, dim). lookup_fn(ids[int64]) -> float32 [len(ids), dim]."""
    meta = _find_meta(table_path, meta_arg)
    vocab, dim = int(meta["vocab"]), int(meta["dim"])
    kind = meta.get("dtype", "float16")
    if kind == "int8_perrow":
        w = np.memmap(table_path, dtype=np.int8, mode="r", shape=(vocab, dim))
        scale_path = table_path.parent / meta["scale_file"]
        scale = np.fromfile(scale_path, dtype=np.float32).reshape(vocab, 1)

        def lookup(ids: np.ndarray) -> np.ndarray:
            return w[ids].astype(np.float32) * scale[ids]  # per-row dequant
        print(f"table: int8 per-row, vocab={vocab} dim={dim} (weight memmap + per-row scale)")
    else:
        table = np.memmap(table_path, dtype=np.dtype(kind), mode="r", shape=(vocab, dim))

        def lookup(ids: np.ndarray) -> np.ndarray:
            return np.asarray(table[ids], dtype=np.float32)
        print(f"table: {kind}, vocab={vocab} dim={dim} (memory-mapped)")
    return lookup, vocab, dim


def _read_ids(path: Path, seq_len: int) -> np.ndarray:
    data = path.read_bytes()
    for dt in (np.int32, np.int64):
        if len(data) == seq_len * np.dtype(dt).itemsize:
            return np.frombuffer(data, dtype=dt).astype(np.int64).reshape(seq_len)
    raise ValueError(f"{path}: {len(data)} bytes != {seq_len} int32/int64")


def _parse_dual_input_list(input_dir: Path) -> list[dict[str, Path]]:
    rows: list[dict[str, Path]] = []
    for line in (input_dir / "input_list.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        entry: dict[str, Path] = {}
        for tok in line.split():
            if ":=" not in tok:
                continue
            name, val = tok.split(":=", 1)
            p = Path(val)
            entry[name] = p if p.is_absolute() else input_dir / p
        if entry:
            rows.append(entry)
    return rows


def _read_source_manifest(input_dir: Path) -> list[dict[str, str]]:
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pid_from_stem(stem: str) -> int:
    match = re.search(r"(?:^|_)pid(-?\d+)(?:_|$)", stem)
    return int(match.group(1)) if match else -1


def _tokenize_captions(captions: list[str], model_dir: Path, seq_len: int):
    """Optional on-board tokenization using the project tokenizer (needs transformers)."""
    import yaml
    from omegaconf import OmegaConf

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from msiglip.utils.tokenizer_utils import get_tokenizer

    cfg = OmegaConf.create(yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8")))
    tok = get_tokenizer(cfg.tokenizer)
    ids_list, mask_list = [], []
    for cap in captions:
        enc = tok(cap, truncation=True, padding="max_length", max_length=seq_len,
                  return_attention_mask=True, return_tensors="np")
        ids_list.append(np.asarray(enc["input_ids"], dtype=np.int64).reshape(seq_len))
        mask_list.append(np.asarray(enc["attention_mask"], dtype=np.float32).reshape(seq_len))
    return ids_list, mask_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="On-board split-text CPU step: tokenize + embedding lookup.")
    p.add_argument("--table", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/bin/token_embedding_v8/token_embedding_int8.bin",
                   help="Token-embedding table .bin (int8 per-row or fp16/fp32). Meta json auto-detected.")
    p.add_argument("--meta", type=Path, help="Override meta json (else auto-detected next to --table).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write inputs_embeds/attention_mask raw + input_list.txt + manifest.csv.")
    p.add_argument("--seq-len", type=int, default=64)
    # Mode A: pre-tokenized input_ids
    p.add_argument("--input-dir", type=Path,
                   help="Text input dir with input_ids/attention_mask raw + input_list.txt (lookup mode).")
    # Mode B: tokenize captions on board (needs transformers + project tokenizer)
    p.add_argument("--captions-file", type=Path,
                   help="UTF-8 file, one caption per line (tokenize-on-board mode).")
    p.add_argument("--pids-file", type=Path, help="Optional: one pid per line, matching --captions-file.")
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8",
                   help="Model dir with config.yaml for the tokenizer (tokenize mode only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_arg = args.meta.expanduser().resolve() if args.meta else None
    lookup, vocab, dim = _build_lookup(args.table.expanduser().resolve(), meta_arg)

    # Resolve per-sample (ids, mask, pid, caption).
    samples: list[dict] = []
    if args.captions_file:  # Mode B — tokenize on board
        captions = [ln.rstrip("\n") for ln in args.captions_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        pids = None
        if args.pids_file:
            pids = [int(x) for x in args.pids_file.read_text().split()]
        ids_list, mask_list = _tokenize_captions(captions, args.model_dir.expanduser().resolve(), args.seq_len)
        for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
            samples.append({"stem": f"{i:05d}", "ids": ids, "mask": mask,
                            "pid": pids[i] if pids else -1, "caption": captions[i][:120]})
    elif args.input_dir:  # Mode A — pre-tokenized input_ids
        input_dir = args.input_dir.expanduser().resolve()
        rows = _parse_dual_input_list(input_dir)
        manifest_rows = _read_source_manifest(input_dir)
        for i, r in enumerate(rows):
            ids = _read_ids(r["input_ids"], args.seq_len)
            mask = np.fromfile(r["attention_mask"], dtype=np.float32).reshape(args.seq_len)
            stem = r["input_ids"].stem.replace("_input_ids", "")
            source_row = manifest_rows[i] if i < len(manifest_rows) else {}
            samples.append({
                "stem": stem,
                "ids": ids,
                "mask": mask,
                "pid": int(source_row.get("pid") or _pid_from_stem(stem)),
                "caption": source_row.get("caption", "")[:120],
            })
    else:
        raise SystemExit("Provide either --input-dir (pre-tokenized) or --captions-file (tokenize on board).")

    lines, manifest = [], []
    for i, s in enumerate(samples):
        ids = np.clip(s["ids"], 0, vocab - 1)
        # The deployable lookup: read 64 rows from the (rotated) table, dequant to float32 for the bin.
        embeds = lookup(ids).astype(np.float32).reshape(1, args.seq_len, dim)
        ep = raw_dir / f"{s['stem']}_inputs_embeds.raw"
        mp = raw_dir / f"{s['stem']}_attention_mask.raw"
        embeds.tofile(ep)
        s["mask"].astype(np.float32).tofile(mp)
        lines.append(f"inputs_embeds:=raw/{ep.name} attention_mask:=raw/{mp.name}")
        manifest.append({"sample_index": i, "pid": s["pid"], "caption": s["caption"],
                         "num_tokens": int(s["mask"].sum())})

    (out_dir / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_index", "pid", "caption", "num_tokens"])
        w.writeheader()
        w.writerows(manifest)

    print(f"wrote {len(samples)} inputs_embeds -> {out_dir}")
    print("next: cd into output dir and run qnn-net-run with text_encoder_split.bin "
          "(--input_list input_list.txt).")


if __name__ == "__main__":
    main()

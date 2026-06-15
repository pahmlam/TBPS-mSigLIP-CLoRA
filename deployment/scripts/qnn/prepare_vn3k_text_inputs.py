#!/usr/bin/env python3
"""Prepare real VN3K caption inputs for the QNN text encoder.

Mirrors `prepare_vn3k_vision_inputs.py` but for the text branch. It tokenizes
VN3K captions with the *exact* project tokenizer (`get_tokenizer(config.tokenizer)`)
and the same call the datasets use (`bases.py`):

    tokenizer(caption, truncation=..., padding="max_length",
              return_attention_mask=True)   # max_length = 64

Two inputs are produced per sample (`input_ids`, `attention_mask`), written as
raw little-endian integer tensors of shape [1, 64], plus an `input_list.txt` in
the multi-input qnn-net-run format:

    input_ids:=raw/00000_..._input_ids.raw attention_mask:=raw/00000_..._attention_mask.raw

Use the output for: (a) AI Hub quantize calibration upload, (b) text QDQ-vs-
PyTorch compare, (c) qnn-net-run on the board.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "text"


def _resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "data_captions_vn3k.json").exists():
        return path
    nested = path / "VN3K"
    if (nested / "data_captions_vn3k.json").exists():
        return nested
    raise SystemExit(f"Cannot find data_captions_vn3k.json under {path}.")


def _first_caption(raw_value) -> str | None:
    """Captions are stored as a stringified Python list; return the first one."""
    if isinstance(raw_value, list):
        captions = raw_value
    elif isinstance(raw_value, str):
        try:
            captions = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            captions = [raw_value]
        if not isinstance(captions, list):
            captions = [str(captions)]
    else:
        return None
    captions = [c for c in (str(x).strip() for x in captions) if c]
    return captions[0] if captions else None


def _load_records(dataset_root: Path, split: str) -> list[dict]:
    annos = json.loads((dataset_root / "data_captions_vn3k.json").read_text(encoding="utf-8"))
    records = []
    for anno in annos:
        if split != "all" and anno.get("split", "") != split:
            continue
        caption = _first_caption(anno.get("captions"))
        if not caption:
            continue
        records.append(
            {
                "pid": int(anno["id"]) - 1,
                "split": anno.get("split", ""),
                "file_path": anno.get("file_path", ""),
                "caption": caption,
            }
        )
    if not records:
        raise SystemExit(f"No VN3K captions found for split={split!r}.")
    return records


def _select(records: list[dict], num: int, selection: str, seed: int, start: int) -> list[dict]:
    if selection == "random":
        rng = random.Random(seed)
        return rng.sample(records, min(num, len(records)))
    return records[start : start + num]


def _load_tokenizer(model_dir: Path):
    import yaml
    from omegaconf import OmegaConf

    from msiglip.utils.tokenizer_utils import get_tokenizer

    config = OmegaConf.create(yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8")))
    return get_tokenizer(config.tokenizer), config


def _np_dtype(name: str) -> np.dtype:
    return {"int64": np.int64, "int32": np.int32}[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare VN3K caption inputs for the QNN text encoder.")
    p.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "VN3K")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10")
    p.add_argument("--split", choices=["train", "test", "validate", "all"], default="test")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--selection", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=2400)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--id-dtype", choices=["int64", "int32"], default="int64",
                   help="Raw integer dtype. int64 matches the ONNX export; switch to int32 if the compiled QNN graph expects it.")
    p.add_argument("--path-mode", choices=["relative", "absolute"], default="relative")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive.")

    dataset_root = _resolve_dataset_root(args.dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dtype = _np_dtype(args.id_dtype)

    tokenizer, _ = _load_tokenizer(args.model_dir.expanduser().resolve())
    records = _load_records(dataset_root, args.split)
    selected = _select(records, args.num_samples, args.selection, args.seed, args.start_index)

    prepared: list[dict] = []
    input_list_lines: list[str] = []
    for idx, rec in enumerate(selected):
        enc = tokenizer(
            rec["caption"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_ids = np.asarray(enc["input_ids"], dtype=dtype).reshape(1, args.max_length)
        attn = np.asarray(enc["attention_mask"], dtype=dtype).reshape(1, args.max_length)

        stem = f"{idx:05d}_pid{rec['pid']}_{_sanitize(Path(rec['file_path']).stem)}"
        ids_path = raw_dir / f"{stem}_input_ids.raw"
        attn_path = raw_dir / f"{stem}_attention_mask.raw"
        input_ids.tofile(ids_path)   # little-endian on x86/arm64
        attn.tofile(attn_path)

        def listed(pth: Path) -> str:
            return str(pth.resolve()) if args.path_mode == "absolute" else str(pth.relative_to(output_dir))

        input_list_lines.append(f"input_ids:={listed(ids_path)} attention_mask:={listed(attn_path)}")
        prepared.append(
            {
                "sample_index": idx,
                "pid": rec["pid"],
                "split": rec["split"],
                "dataset_file_path": rec["file_path"],
                "caption": rec["caption"],
                "input_ids_raw": str(ids_path),
                "attention_mask_raw": str(attn_path),
                "num_tokens": int(attn.sum()),
            }
        )

    (output_dir / "input_list.txt").write_text("\n".join(input_list_lines) + "\n", encoding="utf-8")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "pid", "split", "dataset_file_path", "caption",
                        "input_ids_raw", "attention_mask_raw", "num_tokens"],
        )
        writer.writeheader()
        writer.writerows(prepared)

    print(f"Prepared {len(prepared)} VN3K caption inputs ({args.id_dtype}, max_length={args.max_length})")
    print(f"Output dir:  {output_dir}")
    print(f"Input list:  {output_dir / 'input_list.txt'}")
    print(f"Sample caption[0]: {prepared[0]['caption'][:80]!r} ({prepared[0]['num_tokens']} tokens)")


if __name__ == "__main__":
    main()

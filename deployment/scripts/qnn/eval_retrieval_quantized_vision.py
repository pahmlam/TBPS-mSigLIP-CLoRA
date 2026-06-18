#!/usr/bin/env python3
"""Measure VN3K T2I/I2T R@1 for the quantized vision and/or text encoder.

This is the *decisive* deployment gate: the QDQ cosine proxy (0.90) only tells us
the per-sample embedding drift, not whether retrieval still works. This script runs
the real VN3K test retrieval and reports up to four combos so each encoder's
quantization is isolated and the real end-to-end number is measured:

    - baseline_fp32 : text FP32  + image FP32   (must reproduce ~52.28)
    - vision_int8   : text FP32  + image QDQ    (vision-encoder quantization only)
    - text_int8     : text QDQ   + image FP32   (text-encoder quantization only)
    - both_int8     : text QDQ   + image QDQ    (the real both-INT8 deploy number)

Vision QDQ comes from `--qdq-onnx` (skip with --skip-vision-qdq); text QDQ from
`--text-qdq-onnx` (omit to keep text FP32, the original vision-only behavior).
Image embeddings (FP32 vs QDQ) are run on the SAME preprocessed tensors, and text
embeddings likewise on the SAME token tensors, so each drop isolates quantization.

Retrieval matches `lightning_models.LitTBPS._compute_metrics` *exactly*: features are
the RAW pooler outputs (no L2 normalization) and similarity is a plain dot product,
because the published 52.28 VN3K val_score came from that path. (The separate
`metrics.Evaluator` class normalizes first; do not confuse the two — normalized
ranking gives different numbers and would not reproduce 52.28.)

Preprocessing/tokenization come from the exact training dataset classes
(`VN3K_VI`, `ImageDataset`, `TextDataset`, `get_tokenizer`) driven by the already
resolved `exported_model/config.yaml`, so the FP32 baseline must reproduce ~52.28.
The very same preprocessed image tensor is fed to both PyTorch and ONNX, so the
only difference between the two image branches is the quantization itself.

Deploy gate: T2I R@1 >= 48.0 (FP32 baseline is 52.28).

Example
-------
    venv/bin/python deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
        --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_v2/job_jpr9zro9p_qdq_onnx \
        --json artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_r1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# compare_qnn_with_pytorch lives next to this file; reuse its proven loaders.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_qnn_with_pytorch import _load_pytorch_model, _register_omegaconf_resolvers

from msiglip.data.bases import ImageDataset, TextDataset
from msiglip.data.vn3k_vi import VN3K_VI
from msiglip.utils.metrics import rank
from msiglip.utils.tokenizer_utils import get_tokenizer

DEFAULT_GATE_R1 = 48.0
FP32_BASELINE_R1 = 52.28
_NP_BY_ORT = {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32}


def _resolve_onnx_path(path: Path) -> Path:
    """Accept either a .onnx file or a directory holding exactly one .onnx."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.onnx"))
        if len(candidates) != 1:
            raise SystemExit(f"Expected one .onnx in {path}, found {candidates}")
        return candidates[0]
    return path


def _make_session(onnx_path: Path, ort):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _try_tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _load_config(model_dir: Path):
    import yaml
    from omegaconf import OmegaConf

    _register_omegaconf_resolvers()
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return OmegaConf.create(yaml.safe_load(f))


def _build_test_sets(config, dataset_root: Path, tokenizer):
    """Replicate TBPSDataModule.setup('test') test sets without Lightning."""
    seed = int(config.get("seed", 2400))
    dataset = VN3K_VI(root=str(dataset_root), seed=seed)

    img_size = tuple(config.aug.img.size)
    mean = tuple(config.aug.img.mean)
    std = tuple(config.aug.img.std)

    test_img_set = ImageDataset(
        dataset=dataset.test,
        image_size=img_size,
        is_train=False,
        mean=mean,
        std=std,
    )
    test_txt_set = TextDataset(
        dataset=dataset.test,
        tokenizer=tokenizer,
    )
    return test_img_set, test_txt_set


@torch.no_grad()
def _encode_text(model, loader, device, max_captions: int, session=None):
    """FP32 PyTorch `encode_text`, plus (optional) the text QDQ ONNX on the SAME
    token tensors. Returns (text_fp32, text_q_or_None, pids, onnx_seconds)."""
    name_dtype: dict[str, type] = {}
    out_name = None
    if session is not None:
        for inp in session.get_inputs():
            name_dtype[inp.name] = _NP_BY_ORT.get(inp.type, np.int64)
        out_name = session.get_outputs()[0].name

    feats, q_feats, pids = [], [], []
    seen = 0
    onnx_seconds = 0.0
    for batch in _try_tqdm(loader, desc="text  (FP32+QDQ)" if session else "text  (FP32)"):
        ids_t = batch["caption_input_ids"]
        attn_t = batch["caption_attention_mask"]
        caption_input = {"input_ids": ids_t.to(device), "attention_mask": attn_t.to(device)}
        feat = model.encode_text(caption_input).float().cpu()
        feats.append(feat)

        if session is not None:
            ids_np = ids_t.cpu().numpy()
            attn_np = attn_t.cpu().numpy()
            for i in range(ids_np.shape[0]):
                onnx_inputs = {
                    "input_ids": ids_np[i : i + 1].astype(name_dtype.get("input_ids", np.int64)),
                    "attention_mask": attn_np[i : i + 1].astype(name_dtype.get("attention_mask", np.int64)),
                }
                t0 = time.perf_counter()
                out = session.run([out_name], onnx_inputs)[0]
                onnx_seconds += time.perf_counter() - t0
                q_feats.append(torch.from_numpy(np.asarray(out, dtype=np.float32)).reshape(1, -1))

        pids.append(batch["pids"].view(-1).cpu())
        seen += feat.shape[0]
        if max_captions and seen >= max_captions:
            break

    feats = torch.cat(feats, 0)
    pids = torch.cat(pids, 0)
    text_q = torch.cat(q_feats, 0) if session is not None else None
    if max_captions:
        feats, pids = feats[:max_captions], pids[:max_captions]
        if text_q is not None:
            text_q = text_q[:max_captions]
    return feats, text_q, pids, onnx_seconds


@torch.no_grad()
def _encode_images(model, session, loader, device, image_size, max_images: int):
    """FP32 PyTorch `encode_image`, plus (optional) the vision QDQ ONNX on the SAME
    preprocessed tensors. Returns (image_fp32, image_q_or_None, pids, onnx_seconds)."""
    in_name = out_name = None
    if session is not None:
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name

    fp32_feats, q_feats, pids = [], [], []
    seen = 0
    onnx_seconds = 0.0
    for batch in _try_tqdm(loader, desc="image (FP32+QDQ)" if session else "image (FP32)"):
        images = batch["images"].to(device)
        fp32 = model.encode_image(images).float().cpu()

        if session is not None:
            # ONNX graph has fixed batch=1 input; run one image at a time.
            np_images = batch["images"].cpu().numpy().astype(np.float32, copy=False)
            for i in range(np_images.shape[0]):
                single = np_images[i : i + 1]  # (1, 3, H, W)
                t0 = time.perf_counter()
                out = session.run([out_name], {in_name: single})[0]
                onnx_seconds += time.perf_counter() - t0
                q_feats.append(torch.from_numpy(np.asarray(out, dtype=np.float32)).reshape(1, -1))

        fp32_feats.append(fp32)
        pids.append(batch["pids"].view(-1).cpu())
        seen += fp32.shape[0]
        if max_images and seen >= max_images:
            break

    fp32_feats = torch.cat(fp32_feats, 0)
    pids = torch.cat(pids, 0)
    image_q = torch.cat(q_feats, 0) if session is not None else None
    if max_images:
        fp32_feats, pids = fp32_feats[:max_images], pids[:max_images]
        if image_q is not None:
            image_q = image_q[:max_images]
    return fp32_feats, image_q, pids, onnx_seconds


def _metrics(text_feats, image_feats, text_ids, image_ids, normalize: bool):
    """Mirror LitTBPS._compute_metrics: raw dot product unless normalize=True."""
    import torch.nn.functional as F

    tf = F.normalize(text_feats, dim=1, p=2) if normalize else text_feats
    imf = F.normalize(image_feats, dim=1, p=2) if normalize else image_feats

    t2i_sim = torch.matmul(tf, imf.t())
    i2t_sim = t2i_sim.t()

    t2i_cmc, t2i_map, t2i_minp, _ = rank(
        similarity=t2i_sim, q_pids=text_ids, g_pids=image_ids, max_rank=10, get_mAP=True
    )
    i2t_cmc, i2t_map, i2t_minp, _ = rank(
        similarity=i2t_sim, q_pids=image_ids, g_pids=text_ids, max_rank=10, get_mAP=True
    )
    return {
        "t2i": {
            "R1": t2i_cmc[0].item(),
            "R5": t2i_cmc[4].item(),
            "R10": t2i_cmc[9].item(),
            "mAP": t2i_map.item(),
            "mINP": t2i_minp.item(),
        },
        "i2t": {
            "R1": i2t_cmc[0].item(),
            "R5": i2t_cmc[4].item(),
            "R10": i2t_cmc[9].item(),
            "mAP": i2t_map.item(),
            "mINP": i2t_minp.item(),
        },
    }


def _print_table(title: str, m: dict) -> None:
    table = PrettyTable(["Task", "R1", "R5", "R10", "mAP", "mINP"])
    for task in ("t2i", "i2t"):
        row = [task] + [f"{m[task][k]:.2f}" for k in ("R1", "R5", "R10", "mAP", "mINP")]
        table.add_row(row)
    print(f"\n{title}\n{table}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qdq-onnx",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/runtime/rotated_w8a8_v2/job_jpr9zro9p_qdq_onnx",
        help="QDQ ONNX file or directory (rotated W8A8 vision encoder).",
    )
    parser.add_argument(
        "--text-qdq-onnx",
        type=Path,
        default=None,
        help=(
            "QDQ ONNX file or directory (rotated W8A8 text encoder). When set, also "
            "measures text-isolation (image FP32 + text QDQ) and both-INT8 (image "
            "QDQ + text QDQ) — the real end-to-end deploy number."
        ),
    )
    parser.add_argument(
        "--skip-vision-qdq",
        action="store_true",
        help="Do not run the vision QDQ ONNX (e.g. for a text-isolation-only run).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model",
        help="Merged FP32 model dir (config.yaml + model_fp32.pt) for baseline + text.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing VN3K/. Default: project root.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/mps/cuda).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gate-r1", type=float, default=DEFAULT_GATE_R1, help="T2I R@1 deploy gate."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap gallery images (0=all). Subset metrics are NOT the gate; smoke only.",
    )
    parser.add_argument(
        "--max-captions",
        type=int,
        default=0,
        help="Cap query captions (0=all). Subset metrics are NOT the gate; smoke only.",
    )
    parser.add_argument(
        "--also-cosine",
        action="store_true",
        help="Additionally report normalized (cosine) retrieval as a diagnostic.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_r1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    model_dir = args.model_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    use_vision_qdq = not args.skip_vision_qdq
    vision_onnx_path = _resolve_onnx_path(args.qdq_onnx) if use_vision_qdq else None
    text_onnx_path = _resolve_onnx_path(args.text_qdq_onnx) if args.text_qdq_onnx else None
    if vision_onnx_path is None and text_onnx_path is None:
        raise SystemExit("Nothing to quantize: pass --qdq-onnx and/or --text-qdq-onnx.")

    subset = bool(args.max_images or args.max_captions)
    is_full = not subset

    config = _load_config(model_dir)
    tokenizer = get_tokenizer(config.tokenizer)
    test_img_set, test_txt_set = _build_test_sets(config, dataset_root, tokenizer)
    print(
        f"VN3K test: {len(test_img_set)} gallery images, "
        f"{len(test_txt_set)} query captions"
    )
    print(f"Vision QDQ: {vision_onnx_path or '(skipped)'}")
    print(f"Text QDQ:   {text_onnx_path or '(none — text stays FP32)'}")

    device = torch.device(args.device)
    model, _ = _load_pytorch_model(model_dir, "fp32", str(device))

    vis_session = _make_session(vision_onnx_path, ort) if vision_onnx_path else None
    txt_session = _make_session(text_onnx_path, ort) if text_onnx_path else None

    img_loader = DataLoader(
        test_img_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    txt_loader = DataLoader(
        test_txt_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    text_fp32, text_q, text_ids, text_onnx_s = _encode_text(
        model, txt_loader, device, args.max_captions, session=txt_session
    )
    image_fp32, image_q, image_ids, img_onnx_s = _encode_images(
        model, vis_session, img_loader, device, tuple(config.aug.img.size), args.max_images
    )

    n_img = image_ids.shape[0]
    print(
        f"\nEmbeddings: text {tuple(text_fp32.shape)}, image {tuple(image_fp32.shape)}"
        + (f" | vision ONNX {img_onnx_s:.1f}s ({1000 * img_onnx_s / max(n_img, 1):.1f} ms/img)" if image_q is not None else "")
        + (f" | text ONNX {text_onnx_s:.1f}s ({1000 * text_onnx_s / max(text_ids.shape[0], 1):.1f} ms/cap)" if text_q is not None else "")
    )

    # --- Retrieval combo matrix (raw dot product, matches _compute_metrics) ---
    def _m(tf, imf):
        return _metrics(tf, imf, text_ids, image_ids, normalize=False)

    combos: dict[str, dict] = {"baseline_fp32": _m(text_fp32, image_fp32)}
    _print_table("FP32 BASELINE (text FP32 + image FP32)", combos["baseline_fp32"])

    if image_q is not None:
        combos["vision_int8"] = _m(text_fp32, image_q)
        _print_table("VISION-INT8 (text FP32 + image QDQ)", combos["vision_int8"])
    if text_q is not None:
        combos["text_int8"] = _m(text_q, image_fp32)
        _print_table("TEXT-INT8 (text QDQ + image FP32)", combos["text_int8"])
    if image_q is not None and text_q is not None:
        combos["both_int8"] = _m(text_q, image_q)
        _print_table("BOTH-INT8 (text QDQ + image QDQ) — real deploy number", combos["both_int8"])

    # The deploy combo = the most-quantized available (both > vision > text).
    deploy_key = (
        "both_int8" if "both_int8" in combos
        else "vision_int8" if "vision_int8" in combos
        else "text_int8"
    )
    deploy = combos[deploy_key]
    base = combos["baseline_fp32"]
    t2i_drop = base["t2i"]["R1"] - deploy["t2i"]["R1"]
    i2t_drop = base["i2t"]["R1"] - deploy["i2t"]["R1"]
    print(
        f"\nDeploy combo = {deploy_key}. R@1 drop vs FP32  "
        f"T2I: {base['t2i']['R1']:.2f} -> {deploy['t2i']['R1']:.2f} (-{t2i_drop:.2f})   "
        f"I2T: {base['i2t']['R1']:.2f} -> {deploy['i2t']['R1']:.2f} (-{i2t_drop:.2f})"
    )

    summary = {
        "vision_onnx_model": str(vision_onnx_path) if vision_onnx_path else None,
        "text_onnx_model": str(text_onnx_path) if text_onnx_path else None,
        "model_dir": str(model_dir),
        "num_gallery_images": int(n_img),
        "num_query_captions": int(text_ids.shape[0]),
        "subset_smoke": subset,
        "similarity": "raw_dot_product",
        "combos": combos,
        "deploy_combo": deploy_key,
        "t2i_r1_drop": t2i_drop,
        "i2t_r1_drop": i2t_drop,
        "gate_r1": args.gate_r1,
        "vision_onnx_ms_per_image": 1000 * img_onnx_s / max(n_img, 1) if image_q is not None else None,
        "text_onnx_ms_per_caption": 1000 * text_onnx_s / max(text_ids.shape[0], 1) if text_q is not None else None,
    }

    # Sanity: on the FULL set the FP32 baseline must reproduce ~52.28.
    if is_full:
        baseline_ok = abs(base["t2i"]["R1"] - FP32_BASELINE_R1) <= 1.5
        summary["baseline_sanity_pass"] = baseline_ok
        if not baseline_ok:
            print(
                f"\n[!] BASELINE SANITY FAIL: T2I R@1 {base['t2i']['R1']:.2f} not "
                f"within 1.5 of expected {FP32_BASELINE_R1}. Pipeline likely wrong; "
                "do not trust the quantized number yet."
            )
        else:
            print(f"\n[ok] Baseline sanity: T2I R@1 {base['t2i']['R1']:.2f} ~= {FP32_BASELINE_R1}")

    if args.also_cosine:
        cos = {"baseline_fp32": _metrics(text_fp32, image_fp32, text_ids, image_ids, normalize=True)}
        _print_table("FP32 BASELINE (cosine / normalized, diagnostic)", cos["baseline_fp32"])
        if "both_int8" in combos:
            cos["both_int8"] = _metrics(text_q, image_q, text_ids, image_ids, normalize=True)
            _print_table("BOTH-INT8 (cosine / normalized, diagnostic)", cos["both_int8"])
        summary["combos_cosine"] = cos

    gate_pass = deploy["t2i"]["R1"] >= args.gate_r1
    summary["gate_pass"] = gate_pass
    verdict = "PASS" if gate_pass else "FAIL"
    if subset:
        verdict += " (SUBSET SMOKE — not the real gate; rerun full)"
    print(
        f"\nDEPLOY GATE ({deploy_key}) T2I R@1 >= {args.gate_r1}: "
        f"{deploy['t2i']['R1']:.2f} -> {verdict}"
    )

    out = args.json.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

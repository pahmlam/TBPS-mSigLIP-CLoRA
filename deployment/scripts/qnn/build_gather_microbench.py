#!/usr/bin/env python3
"""A1/A2 — Build a minimal ``Gather`` microbenchmark to isolate the text-board bug.

The full text encoder is ignored by HTP at runtime: board output is bit-identical
for real vs zero ``input_ids`` (see [deploy-master] §11). Four hypotheses remain
(§12.2): H1 = HTP cannot lower a dynamic ``Gather(weight, input_ids)``; H2 = the
~192 MB INT8 token table exceeds an HTP/DLC constant-buffer limit; H3 = the AI Hub
quantize pass breaks Gather; H4 = a ``qnn-net-run`` input-binding mistake.

This builds the smallest possible graph that exercises *only* the embedding lookup:

    input_ids [1, L] int32  ->  Gather(W, axis=0)  ->  embeds [1, L, 768]

It emits two variants that differ in **exactly one variable — table size**:

* ``small`` : vocab ~1000 (first rows of the real table). ~3 MB INT8 constant.
* ``big``   : the full ~250k x 768 table from the failing model. ~192 MB INT8 constant.

Decision (run each on board, real vs zero ids):
* small PASS & big FAIL  -> H2 (constant-buffer size limit) — the cleanest result.
* small FAIL             -> H1/H3 (Gather itself, any size).
* both PASS              -> Gather is NOT the culprit; the full-graph fault is elsewhere.

For each variant it also writes matched ``input_ids`` raw sets (real + zero, indices
valid for that vocab) and an ``input_list.txt``, plus a local ONNX Runtime sanity
check confirming the ONNX is correct (real != zero) BEFORE spending an AI Hub job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from compare_qnn_with_pytorch import _load_pytorch_model  # noqa: E402

EMBED_DIM = 768


def _build_gather_onnx(weight: np.ndarray, onnx_path: Path, seq_len: int) -> None:
    """input_ids [1, L] int32 -> Gather(weight, axis=0) -> embeds [1, L, D]."""
    vocab, dim = weight.shape
    w_init = numpy_helper.from_array(weight.astype(np.float32), name="token_table")
    inp = helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, seq_len])
    out = helper.make_tensor_value_info("embeds", TensorProto.FLOAT, [1, seq_len, dim])
    gather = helper.make_node("Gather", ["token_table", "input_ids"], ["embeds"], axis=0)
    graph = helper.make_graph([gather], f"gather_microbench_v{vocab}", [inp], [out], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=False)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    data_name = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model, str(onnx_path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=data_name, size_threshold=1024, convert_attribute=False,
    )


def _write_inputs(out_dir: Path, real_ids: np.ndarray, vocab: int) -> None:
    """Write a real and a zero input_ids set (int32) valid for `vocab`."""
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    real_ids = np.mod(real_ids.astype(np.int64), vocab).astype(np.int32)  # keep < vocab
    zero_ids = np.zeros_like(real_ids)
    lines_real, lines_zero = [], []
    for i in range(real_ids.shape[0]):
        rp = raw / f"{i:05d}_real_input_ids.raw"
        zp = raw / f"{i:05d}_zero_input_ids.raw"
        real_ids[i:i + 1].tofile(rp)
        zero_ids[i:i + 1].tofile(zp)
        lines_real.append(f"input_ids:=raw/{rp.name}")
        lines_zero.append(f"input_ids:=raw/{zp.name}")
    (out_dir / "input_list_real.txt").write_text("\n".join(lines_real) + "\n", encoding="utf-8")
    (out_dir / "input_list_zero.txt").write_text("\n".join(lines_zero) + "\n", encoding="utf-8")


def _ort_sanity(onnx_path: Path, real_ids: np.ndarray, vocab: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ids0 = np.mod(real_ids[0:1].astype(np.int64), vocab).astype(np.int32)
    zeros = np.zeros_like(ids0)
    real_out = np.asarray(sess.run(None, {name: ids0})[0], dtype=np.float32)
    zero_out = np.asarray(sess.run(None, {name: zeros})[0], dtype=np.float32)
    rn, zn = np.linalg.norm(real_out.reshape(-1)), np.linalg.norm(zero_out.reshape(-1))
    cos = float(real_out.reshape(-1) @ zero_out.reshape(-1) / (rn * zn + 1e-9))
    return {"real_zero_cosine": cos, "real_norm": float(rn), "zero_norm": float(zn),
            "depends_on_ids": bool(not np.allclose(real_out, zero_out))}


def _load_real_ids(input_dir: Path, seq_len: int) -> np.ndarray:
    raws = sorted(input_dir.glob("raw/*_input_ids.raw"))
    if not raws:
        raise SystemExit(f"No *_input_ids.raw under {input_dir}/raw")
    rows = []
    for p in raws:
        data = p.read_bytes()
        for dt in (np.int32, np.int64):
            if len(data) == seq_len * np.dtype(dt).itemsize:
                rows.append(np.frombuffer(data, dtype=dt).astype(np.int64).reshape(1, seq_len))
                break
    return np.concatenate(rows, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Gather microbench ONNX (small + big table).")
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/exports/gather_microbench")
    p.add_argument("--real-input-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--small-vocab", type=int, default=1000)
    p.add_argument("--variants", nargs="+", choices=["small", "big"], default=["small", "big"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()

    print(f"Loading token table from {args.model_dir} ...")
    model, _ = _load_pytorch_model(args.model_dir.expanduser().resolve(), "fp32", "cpu")
    table = model.text_model.embeddings.token_embedding.weight.detach().float().cpu().numpy()
    vocab_full, dim = table.shape
    print(f"token table: vocab={vocab_full}, dim={dim} (FP32 ~{table.nbytes / 1e6:.0f} MB)")

    real_ids = _load_real_ids(args.real_input_dir.expanduser().resolve(), args.seq_len)
    print(f"real input_ids: {real_ids.shape} from {args.real_input_dir}")

    plans = {
        "small": (min(args.small_vocab, vocab_full), out_dir / "small"),
        "big": (vocab_full, out_dir / "big"),
    }
    for variant in args.variants:
        vocab, vdir = plans[variant]
        onnx_path = vdir / f"gather_{variant}.onnx"
        print(f"\n=== variant '{variant}': vocab={vocab} (INT8 ~{vocab * dim / 1e6:.0f} MB) ===")
        _build_gather_onnx(table[:vocab], onnx_path, args.seq_len)
        _write_inputs(vdir, real_ids, vocab)
        sanity = _ort_sanity(onnx_path, real_ids, vocab)
        print(f"  onnx: {onnx_path}")
        print(f"  ORT sanity (real vs zero ids): depends_on_ids={sanity['depends_on_ids']}, "
              f"cosine={sanity['real_zero_cosine']:.4f}")
        if not sanity["depends_on_ids"]:
            print("  [WARN] ORT output does NOT depend on ids — the ONNX itself is wrong, fix before AI Hub.")

    print(
        "\nNext (user, AI Hub + board):\n"
        "  - quantize/compile/link each variant; input_ids is an INT32 INDEX (do NOT quantize it).\n"
        "  - on board: run input_list_real.txt and input_list_zero.txt, compare outputs.\n"
        "  - small PASS & big FAIL => H2 (size). small FAIL => H1/H3. both PASS => Gather not the culprit."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B1 — Export the SPLIT text encoder to ONNX opset-20.

Split-text deployment path (see [deploy-master] §11–§12). The full text encoder
fails on HTP v68 because the dynamic ``Gather(token_embedding.weight, input_ids)``
is ignored at runtime (board output is bit-identical for real vs zero token IDs).
The fix is to move the embedding lookup off the HTP and keep the heavy transformer
on the NPU:

    host/CPU   :  inputs_embeds = token_embedding[input_ids]      # [1, L, 768]
    HTP graph  :  inputs_embeds + attention_mask
                  -> (+ position_embedding) -> encoder
                  -> final_layer_norm -> last-token pooler -> head -> text_projection

This script exports ONLY the HTP graph (the part after the token lookup). The token
table is *not* part of this graph, so the giant 250k x 768 dynamic Gather is gone.
``SiglipTextEmbeddings`` already supports ``inputs_embeds`` (it skips the token
lookup but still adds the small position embedding internally), so the host side
only needs the raw token lookup — no position add.

Works on the rotated model (``exported_model_text_rotated_learned_qat_v8``): the
mean-preserving ``Q`` is already folded into both the token and position embeddings,
so host-side ``token_embedding[input_ids]`` already lives in the rotated space and
the in-graph position add is the rotated position table — consistent end to end.

GATE (B1): with ``--check-input-dir``, the split path must reproduce the full
``encode_text`` PyTorch reference with cosine ~1.0 (the embedding lookup is exact).

Example:
    venv/bin/python deployment/scripts/qnn/export_split_text_onnx.py \
      --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
      --attention-mask-dtype float32 \
      --check-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32 \
      --json /tmp/split_text_static.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import onnx  # noqa: E402

from compare_qnn_with_pytorch import _load_pytorch_model, _vector_stats, _write_json  # noqa: E402

# Finite additive-mask constant (matches the full-graph link-safe deploy form, §12A.4-5).
# exp(-32) ~ 1.3e-14, so masked-position leakage is negligible vs FP32/INT8 noise.
MASK_VALUE = -32.0


class SplitTextWrapper(nn.Module):
    """Text transformer + head that consumes ``inputs_embeds`` (post token lookup).

    Mirrors ``SiglipTextTransformer.forward`` exactly, except the token-embedding
    Gather is removed: ``inputs_embeds`` is supplied by the host. The position
    embedding (small, static) stays in the graph via ``embeddings(inputs_embeds=...)``.
    """

    def __init__(self, tbps_model: nn.Module) -> None:
        super().__init__()
        self.tbps = tbps_model
        self.text_model = tbps_model.text_model  # SiglipTextTransformer

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tm = self.text_model
        # inputs_embeds: [B, L, 768] = token_embedding[input_ids] (host side, no position).
        hidden_states = tm.embeddings(inputs_embeds=inputs_embeds)  # + position internally
        # Link-safe finite additive mask built directly (NOT _prepare_4d_attention_mask):
        # the HF helper emits an Expand->Cast(FLOAT)->Sub->Cast(BOOL)->Where island plus a
        # -FLT_MAX sentinel; AI Hub materializes the cast as an internal float tensor that
        # HTP v68 rejects at link (/Cast_output_0_updated). Building M = (1-mask)*(-32)
        # expanded to [B,1,L,L] gives the same attention semantics with no Cast/Where/-FLT_MAX.
        # attention_mask is a float32 graph input already — do NOT add a `.to(dtype)` cast:
        # a redundant float32->float32 Cast materializes as /Cast_output_0, which AI Hub turns
        # into an internal float tensor (/Cast_output_0_updated) that HTP v68 rejects at link.
        bsz, seq_len = attention_mask.shape[0], attention_mask.shape[1]
        # Reshape the input mask to [B,1,1,L] FIRST, then Sub/Mul, so after the Expand is
        # stripped (_strip_mask_expand) the graph is Reshape->Sub->Mul->Add — structurally
        # identical to the proven full-graph link-safe form (no float op between Mul and Add).
        m = attention_mask.view(bsz, 1, 1, seq_len)  # [B,1,1,L]
        add = (1.0 - m) * MASK_VALUE  # [B,1,1,L]: 0 where valid, -32 where padded
        mask4d = add.expand(bsz, 1, seq_len, seq_len)  # [B,1,L,L] (Expand stripped post-export)
        encoder_outputs = tm.encoder(
            inputs_embeds=hidden_states,
            attention_mask=mask4d,
            return_dict=False,
        )
        last_hidden = tm.final_layer_norm(encoder_outputs[0])
        pooled = last_hidden[:, -1, :]  # sticky-EOS last token
        pooled = tm.head(pooled)
        if hasattr(self.tbps.backbone, "text_projection"):
            pooled = self.tbps.backbone.text_projection(pooled)
        return pooled  # [B, 768]


def _token_lookup(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Host-side raw token embedding (no position add): [1, L, 768]."""
    return model.text_model.embeddings.token_embedding(input_ids)


def _strip_mask_expand(onnx_path: Path) -> dict[str, object]:
    """Remove the additive-mask Expand so the [B,1,1,L] mask broadcasts inside attention.

    PyTorch's SiglipAttention asserts the mask is [B,1,L,L], so the wrapper must build
    (and the trace must materialize) an Expand to [B,1,L,L]. But AI Hub treats that
    materialized float mask as a floating coefficient tensor (`/Expand_coef`) that HTP v68
    rejects at link. The non-causal text attention `attn_weights[B,H,L,L] + mask` broadcasts
    a [B,1,1,L] mask natively, so we rewire the 12 self-attn Add nodes to consume the
    pre-Expand [B,1,1,L] tensor and drop the Expand. (Same idea as patch_text_qnn_link_safe_mask.)
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    consumers: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    stripped = []
    for node in list(graph.node):
        if node.op_type != "Expand":
            continue
        out = node.output[0]
        cons = consumers.get(out, [])
        # Only the additive-mask Expand feeds the self-attention Add nodes.
        if not cons or not all(c.op_type == "Add" and "self_attn/Add" in c.name for c in cons):
            continue
        pre_expand = node.input[0]  # [B,1,1,L] reshaped mask
        for c in cons:
            for k, name in enumerate(c.input):
                if name == out:
                    c.input[k] = pre_expand
        graph.node.remove(node)
        stripped.append({"expand": node.name, "rewired_adds": len(cons), "feed": pre_expand})

    if not stripped:
        raise RuntimeError("No additive-mask Expand found feeding self_attn/Add — graph changed?")
    onnx.save(model, str(onnx_path))
    return {"stripped_expands": stripped}


def _save_external(model: onnx.ModelProto, onnx_path: Path) -> None:
    data_name = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _print_op_counts(onnx_path: Path) -> dict[str, int]:
    m = onnx.load(str(onnx_path), load_external_data=False)
    opsets = [(op.domain or "ai.onnx", op.version) for op in m.opset_import]
    counts = Counter(n.op_type for n in m.graph.node)
    print(f"\nopset imports: {opsets}")
    print(f"total nodes: {sum(counts.values())}")
    for k in ("Gather", "Gelu", "Pow", "Tanh", "LayerNormalization", "ReduceMean", "MatMul", "Softmax"):
        if k in counts:
            print(f"  {k:18s}: {counts[k]}")
    return dict(counts)


def _read_raw(path: Path, length: int) -> tuple[np.ndarray, np.dtype]:
    data = path.read_bytes()
    for dtype in (np.int64, np.int32, np.float32):
        if len(data) == length * np.dtype(dtype).itemsize:
            return np.frombuffer(data, dtype=dtype).reshape(1, length).copy(), np.dtype(dtype)
    raise ValueError(f"{path}: {len(data)} bytes does not match {length} int64/int32/float32 values")


def _parse_dual_input_list(input_dir: Path) -> list[dict[str, Path]]:
    input_list = input_dir / "input_list.txt"
    rows: list[dict[str, Path]] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        entry: dict[str, Path] = {}
        for token in line.split():
            if ":=" not in token:
                continue
            name, value = token.split(":=", 1)
            p = Path(value)
            entry[name] = p if p.is_absolute() else input_dir / p
        if entry:
            rows.append(entry)
    if not rows:
        raise ValueError(f"No input rows found in {input_list}")
    return rows


def _static_gate(
    args: argparse.Namespace,
    model: nn.Module,
    onnx_path: Path,
    seq_len: int,
) -> dict:
    import onnxruntime as ort

    input_dir = args.check_input_dir.expanduser().resolve()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    in_types = {i.name: i.type for i in session.get_inputs()}
    out_name = session.get_outputs()[0].name

    dump_dir = args.dump_embeds_dir.expanduser().resolve() if args.dump_embeds_dir else None
    if dump_dir is not None:
        (dump_dir / "raw").mkdir(parents=True, exist_ok=True)
        dump_lines: list[str] = []

    rows_in = _parse_dual_input_list(input_dir)
    cosines: list[float] = []
    first_row: dict = {}
    with torch.no_grad():
        for idx, entry in enumerate(rows_in):
            ids_np, _ = _read_raw(entry["input_ids"], seq_len)
            # attention_mask is float32; read it explicitly. (int32 and float32 both occupy
            # seq_len*4 bytes, so size-based dtype guessing would misread the mask.)
            attn_np = np.fromfile(entry["attention_mask"], dtype=np.float32).reshape(1, seq_len).copy()
            ids_t = torch.from_numpy(ids_np.astype(np.int64))
            attn_t = torch.from_numpy(attn_np)

            # PyTorch reference: the FULL encode_text path (token lookup in-model).
            ref = model.encode_text(
                {"input_ids": ids_t, "attention_mask": attn_t}
            ).detach().float().cpu().reshape(-1)

            # Split path: host-side token lookup -> split ONNX graph.
            embeds = _token_lookup(model, ids_t).detach().float().cpu().numpy()  # [1,L,768]
            feed = {"inputs_embeds": embeds.astype(np.float32),
                    "attention_mask": attn_np.astype(np.float32)}
            onnx_out = torch.from_numpy(
                np.asarray(session.run([out_name], feed)[0], dtype=np.float32)
            ).reshape(-1)

            cos = torch.dot(F.normalize(onnx_out, dim=0), F.normalize(ref, dim=0)).item()
            cosines.append(cos)
            o_stats = _vector_stats(onnx_out.tolist())
            row = {
                "result_index": idx,
                "cosine_l2": cos,
                "onnx_norm": o_stats["norm"],
                "ref_norm": _vector_stats(ref.tolist())["norm"],
                "onnx_has_nan": o_stats["has_nan"],
            }
            if idx == 0:
                first_row = row

            if dump_dir is not None:
                stem = entry["input_ids"].stem.replace("_input_ids", "")
                emb_path = dump_dir / "raw" / f"{stem}_inputs_embeds.raw"
                mask_path = dump_dir / "raw" / f"{stem}_attention_mask.raw"
                embeds.astype(np.float32).tofile(emb_path)
                attn_np.astype(np.float32).tofile(mask_path)
                dump_lines.append(
                    f"inputs_embeds:=raw/{emb_path.name} attention_mask:=raw/{mask_path.name}"
                )

    if dump_dir is not None:
        (dump_dir / "input_list.txt").write_text("\n".join(dump_lines) + "\n", encoding="utf-8")
        print(f"\nDumped {len(dump_lines)} inputs_embeds .raw for board (B2): {dump_dir}")

    summary = {
        "onnx_model": str(onnx_path),
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "check_input_dir": str(input_dir),
        "num_samples": len(cosines),
        "onnx_input_types": in_types,
        "cosine_l2_mean": sum(cosines) / len(cosines),
        "cosine_l2_min": min(cosines),
        "cosine_l2_max": max(cosines),
        "first_row": first_row,
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export split (inputs_embeds) text encoder to ONNX opset-20.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8",
    )
    p.add_argument("--output-subdir", default="text_onnx_split")
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument(
        "--attention-mask-dtype",
        choices=["int64", "float32"],
        default="float32",
        help="float32 avoids the internal attention_mask Cast(FLOAT) island that QNN HTP link rejects.",
    )
    p.add_argument("--check-input-dir", type=Path,
                   help="If set, run the B1 static gate (split ONNX vs full encode_text) on this input dir.")
    p.add_argument("--dump-embeds-dir", type=Path,
                   help="If set (with --check-input-dir), write inputs_embeds .raw + input_list for the board (B2).")
    p.add_argument("--json", type=Path, help="Write the static-gate summary JSON here.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()

    print(f"Loading model from {model_dir} ...")
    model, _ = _load_pytorch_model(model_dir, "fp32", args.device)
    model.eval()

    out_dir = model_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "text_encoder_split.onnx"

    dummy_embeds = torch.zeros(1, args.seq_len, args.embed_dim, dtype=torch.float32)
    dummy_mask = torch.ones(
        1,
        args.seq_len,
        dtype=torch.float32 if args.attention_mask_dtype == "float32" else torch.long,
    )

    print(f"Exporting split text encoder (inputs_embeds) to ONNX opset {args.opset} ...")
    torch.onnx.export(
        SplitTextWrapper(model),
        (dummy_embeds, dummy_mask),
        str(onnx_path),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "inputs_embeds": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embedding": {0: "batch_size"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    loaded = onnx.load(str(onnx_path), load_external_data=True)
    _save_external(loaded, onnx_path)
    # Strip the materialized [B,1,L,L] mask Expand (HTP rejects it as /Expand_coef float island);
    # the 12 self-attn Add nodes broadcast the [B,1,1,L] mask natively.
    strip_summary = _strip_mask_expand(onnx_path)
    print(f"Saved: {onnx_path}")
    print(f"Mask Expand stripped: {strip_summary['stripped_expands']}")

    counts = _print_op_counts(onnx_path)
    if counts.get("Gather", 0):
        print(
            f"\n[note] {counts['Gather']} Gather op(s) remain — expected: position-embedding "
            "lookup is a static slice, NOT the dynamic 250k token table."
        )

    if args.check_input_dir is not None:
        # IMPORTANT: torch.onnx.export contaminates the in-process model state (its forward
        # output drifts after a trace). Reload a FRESH model so the gate's PyTorch reference
        # is the true encode_text, not the post-export drifted one.
        gate_model, _ = _load_pytorch_model(model_dir, "fp32", args.device)
        gate_model.eval()
        summary = _static_gate(args, gate_model, onnx_path, args.seq_len)
        print("\n=== B1 static gate (split ONNX vs full encode_text) ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.json is not None:
            _write_json(summary, args.json.expanduser().resolve())
        gate = "PASS" if summary["cosine_l2_min"] >= 0.999 else "FAIL"
        print(f"\nB1 gate (cosine min >= 0.999): {gate} "
              f"(mean {summary['cosine_l2_mean']:.8f}, min {summary['cosine_l2_min']:.8f})")


if __name__ == "__main__":
    main()

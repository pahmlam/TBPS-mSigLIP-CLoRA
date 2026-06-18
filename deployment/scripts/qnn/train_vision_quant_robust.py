#!/usr/bin/env python3
"""Quantization-aware fine-tuning by teacher-student distillation (vision OR text).

This script does not export a custom QDQ graph. It fine-tunes the FP32 encoder
(`--modality vision` or `--modality text`) while injecting lightweight fake-quant
noise into the sensitive blocks, then saves a normal FP32 deployment export
directory. The follow-up step is the existing ONNX export + AI Hub native
quantizer on that tuned FP32 model.

The two towers share the SiglipEncoderLayer / SiglipAttention, so the fake-quant
coverage ladder (GELU + residual -> +head -> +linears -> +attention matmuls) and
the EMA per-tensor observer are identical. Only the data path differs: vision
distills `encode_image` on NCHW image .raw inputs; text distills `encode_text`
on int input_ids+attention_mask .raw inputs, with a last-token Linear head.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_qnn_with_pytorch import (  # noqa: E402
    _load_pytorch_model,
    _parse_input_list,
    _read_raw_tensor,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts/deployment/exports/exported_model"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_train_calib_2000"
DEFAULT_VAL_DIR = PROJECT_ROOT / "artifacts/deployment/qnn_inputs/vn3k_test_100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/deployment/exports/exported_model_qat_v1"


class RawVisionDataset(Dataset):
    """Dataset backed by QNN-ready NCHW float32 .raw image tensors."""

    def __init__(
        self,
        input_dir: Path,
        input_list_name: str,
        image_size: int,
        max_samples: int | None = None,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.image_size = image_size
        self.raw_paths = _parse_input_list(self.input_dir, input_list_name)
        if max_samples is not None:
            self.raw_paths = self.raw_paths[:max_samples]
        if not self.raw_paths:
            raise ValueError(f"No raw inputs found in {self.input_dir}")

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return _read_raw_tensor(self.raw_paths[index], self.image_size).squeeze(0)


class RawTextDataset(Dataset):
    """Dataset backed by QNN-ready int `input_ids` + `attention_mask` .raw tensors.

    Each sample is a dict of two integer tensors (shape (seq_len,)); the default
    collate stacks them into a batched dict consumed by `encode_text`.
    """

    def __init__(
        self,
        input_dir: Path,
        input_list_name: str,
        seq_len: int,
        max_samples: int | None = None,
    ) -> None:
        # Lazy import keeps the vision path free of the onnx/onnxruntime deps.
        from compare_text_onnx_with_pytorch import _parse_dual_input_list, _read_raw_ints

        self._read_raw_ints = _read_raw_ints
        self.seq_len = seq_len
        self.rows = _parse_dual_input_list(input_dir.expanduser().resolve(), input_list_name)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No dual int inputs found in {input_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import numpy as np

        entry = self.rows[index]
        ids = self._read_raw_ints(entry["input_ids"], self.seq_len)  # (1, L)
        attn = self._read_raw_ints(entry["attention_mask"], self.seq_len)  # (1, L)
        return {
            "input_ids": torch.from_numpy(ids.astype(np.int64)).squeeze(0),  # (L,)
            "attention_mask": torch.from_numpy(attn.astype(np.int64)).squeeze(0),  # (L,)
        }


def _fake_quant_symmetric(
    x: torch.Tensor,
    bits: int,
    eps: float,
    per_tensor: bool = True,
    max_abs: torch.Tensor | None = None,
) -> torch.Tensor:
    if bits <= 0:
        return x
    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        return x

    if max_abs is None:
        if per_tensor or x.ndim <= 1:
            # One scale for the WHOLE activation tensor. This matches AI Hub's
            # per-tensor W8A8 (the deployed scheme). The legacy per-sample mode
            # below gives each image its own scale -> easier for the student, so
            # the simulated cosine looks great (~0.975) but does not transfer to
            # the real per-tensor quantize (only ~0.92). Per-tensor closes that.
            max_abs = x.detach().abs().max()
        else:
            reduce_dims = tuple(range(1, x.ndim))
            max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=eps)
    quantized = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return x + (quantized - x).detach()


def _make_fq_attention_forward(controller: "FakeQuantController", key: str):
    """Eager SigLIP attention forward with fake-quant on the two internal
    activation-by-activation matmuls (which `nn.Module` forward hooks cannot reach).

    Quantizes the score matrix (Q*K^T), the softmax probabilities, and the context
    (probs*V) --- the activations AI Hub quantizes on-device but `--quant-linears`
    misses. q/k/v/out_proj outputs are still covered by their own linear hooks.
    Mirrors the eager `SiglipAttention.forward`; numerically identical when fake-quant
    is disabled (clean path stays exact).
    """

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = controller._quantize(scores, key + ".scores")
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = controller._quantize(probs, key + ".probs")
        probs = F.dropout(probs, p=self.dropout, training=self.training)

        ctx = torch.matmul(probs, v)
        ctx = controller._quantize(ctx, key + ".context")
        ctx = ctx.transpose(1, 2).contiguous().reshape(bsz, q_len, self.embed_dim)
        out = self.out_proj(ctx)
        return out, (probs if output_attentions else None)

    return forward


class FakeQuantController:
    """Installs activation fake-quant hooks for selected SigLIP encoder blocks.

    `modality` selects the tower: "vision" -> backbone.vision_model (MHA pooling
    head), "text" -> backbone.text_model (final_layer_norm + Linear head). The
    encoder-layer hooks (GELU, residual, linears, attention matmuls) are identical
    across towers because both use the same SiglipEncoderLayer / SiglipAttention.
    """

    def __init__(
        self,
        model: nn.Module,
        start_layer: int,
        end_layer: int,
        bits: int,
        eps: float,
        per_tensor: bool = True,
        observer: str = "ema",
        ema_momentum: float = 0.99,
        quant_head: bool = False,
        quant_linears: bool = False,
        quant_attention: bool = False,
        modality: str = "vision",
    ) -> None:
        self.model = model
        self.modality = modality
        self.tower_name = "vision_model" if modality == "vision" else "text_model"
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.quant_head = quant_head
        self.quant_linears = quant_linears
        self.quant_attention = quant_attention
        self.patched_attn: list[nn.Module] = []
        self.bits = bits
        self.eps = eps
        self.per_tensor = per_tensor
        self.observer = observer
        self.ema_momentum = ema_momentum
        # EMA running per-tensor max_abs per hooked tensor (calibrate-once style).
        self.running_max: dict[str, torch.Tensor] = {}
        self.enabled = True
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.hooked_modules: list[str] = []

    def _tower(self) -> nn.Module:
        return getattr(self.model.backbone, self.tower_name)

    def install(self) -> None:
        tower = self._tower()
        prefix = f"backbone.{self.tower_name}"
        layers = tower.encoder.layers
        if self.start_layer < 0 or self.end_layer >= len(layers):
            raise ValueError(
                f"Layer range {self.start_layer}-{self.end_layer} is invalid for "
                f"{len(layers)} {self.modality} encoder layers."
            )

        for index in range(self.start_layer, self.end_layer + 1):
            layer = layers[index]
            self._hook_module(
                f"{prefix}.encoder.layers.{index}.mlp.activation_fn",
                layer.mlp.activation_fn,
                output_is_tuple=False,
            )
            self._hook_module(
                f"{prefix}.encoder.layers.{index}",
                layer,
                output_is_tuple=True,
            )
            if self.quant_linears:
                # AI Hub quantizes EVERY activation; the two hooks above only cover
                # GELU out + residual. Hook all linear outputs so QAT robustifies
                # the q/k/v/out_proj and fc1/fc2 activations too (faithful coverage).
                for sub in (
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj", "mlp.fc1", "mlp.fc2",
                ):
                    self._hook_module(
                        f"{prefix}.encoder.layers.{index}.{sub}",
                        layer.get_submodule(sub),
                        output_is_tuple=False,
                    )
            if self.quant_attention:
                # Forward hooks cannot reach the two intra-attention matmuls; replace
                # the eager attention forward with a fake-quant-aware copy instead.
                # Same SiglipAttention class for both towers; text passes the 4D
                # attention_mask (added after scores are quantized).
                attn = layer.self_attn
                key = f"{prefix}.encoder.layers.{index}.self_attn"
                attn.forward = types.MethodType(_make_fq_attention_forward(self, key), attn)
                self.patched_attn.append(attn)
                self.hooked_modules.append(key + " [scores/probs/context]")

        if self.quant_head:
            self._install_head_hooks(tower, prefix)

    def _install_head_hooks(self, tower: nn.Module, prefix: str) -> None:
        """Hook the final embedding-producing stage; its INT8 error is not averaged
        out by any later layer.

        Vision: MHA pooling head (post_layernorm -> head.attention -> head.mlp ->
        head). nn.MHA returns (attn_out, weights) -> tuple hook. Text: a plain
        Linear head reading the last-token-pooled final_layer_norm output.
        """
        if self.modality == "vision":
            self._hook_module(
                f"{prefix}.post_layernorm", tower.post_layernorm, output_is_tuple=False
            )
            self._hook_module(
                f"{prefix}.head.attention", tower.head.attention, output_is_tuple=True
            )
            self._hook_module(
                f"{prefix}.head.mlp.activation_fn",
                tower.head.mlp.activation_fn,
                output_is_tuple=False,
            )
            self._hook_module(f"{prefix}.head", tower.head, output_is_tuple=False)
            if self.quant_linears:
                for sub in ("mlp.fc1", "mlp.fc2"):
                    self._hook_module(
                        f"{prefix}.head.{sub}",
                        tower.head.get_submodule(sub),
                        output_is_tuple=False,
                    )
        else:
            self._hook_module(
                f"{prefix}.final_layer_norm", tower.final_layer_norm, output_is_tuple=False
            )
            self._hook_module(f"{prefix}.head", tower.head, output_is_tuple=False)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        # Restore the original (class-level) attention forward on patched modules.
        for attn in self.patched_attn:
            if "forward" in attn.__dict__:
                del attn.__dict__["forward"]
        self.patched_attn.clear()
        self.hooked_modules.clear()

    @contextmanager
    def disabled(self):
        old_value = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old_value

    def _quantize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if not self.enabled or not torch.is_floating_point(value):
            return value
        if self.observer == "ema":
            # Per-tensor scale from an EMA of the running max_abs (one fixed-ish
            # scale per tensor, like AI Hub calibrating once on the calib set).
            # This is the deploy-faithful observer: it closes the sim<->real gap
            # that a per-batch dynamic scale leaves open.
            cur = value.detach().abs().max()
            prev = self.running_max.get(key)
            running = cur if prev is None else (
                self.ema_momentum * prev + (1.0 - self.ema_momentum) * cur
            )
            self.running_max[key] = running.detach()
            return _fake_quant_symmetric(
                value, bits=self.bits, eps=self.eps, max_abs=running
            )
        return _fake_quant_symmetric(
            value, bits=self.bits, eps=self.eps, per_tensor=self.per_tensor
        )

    def _hook_module(self, name: str, module: nn.Module, output_is_tuple: bool) -> None:
        def hook(_module: nn.Module, _inputs: tuple, output):
            if output_is_tuple:
                if not isinstance(output, tuple) or not output:
                    return output
                return (self._quantize(output[0], name),) + output[1:]
            return self._quantize(output, name)

        self.handles.append(module.register_forward_hook(hook))
        self.hooked_modules.append(name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _text_seq_len(config, fallback: int = 64) -> int:
    try:
        return int(config.tokenizer.model_max_length)
    except Exception:
        return fallback


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _freeze_for_qat(
    model: nn.Module,
    modality: str,
    start_layer: int,
    end_layer: int,
    train_projection: bool,
    train_head: bool = False,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    tower_name = "vision_model" if modality == "vision" else "text_model"
    projection_attr = "visual_projection" if modality == "vision" else "text_projection"
    trainable_prefixes = [
        f"backbone.{tower_name}.encoder.layers.{index}."
        for index in range(start_layer, end_layer + 1)
    ]
    if train_projection and hasattr(model.backbone, projection_attr):
        trainable_prefixes.append(f"backbone.{projection_attr}.")
    if train_head:
        trainable_prefixes.append(f"backbone.{tower_name}.head.")
        # The final norm feeding the head: post_layernorm (vision) / final_layer_norm (text).
        final_norm = "post_layernorm" if modality == "vision" else "final_layer_norm"
        trainable_prefixes.append(f"backbone.{tower_name}.{final_norm}.")

    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(f"No trainable parameters selected for {modality} QAT.")
    return trainable_names


def _trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def _batch_cosine(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        F.normalize(student.float(), dim=-1),
        F.normalize(teacher.float(), dim=-1),
        dim=-1,
    )


def _distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = _batch_cosine(student, teacher)
    cosine_loss = 1.0 - cosine.mean()
    mse_loss = F.mse_loss(student.float(), teacher.float())
    return cosine_loss + mse_weight * mse_loss, cosine_loss, mse_loss


def _weighted_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = _batch_cosine(student, teacher)
    cosine_loss = 1.0 - cosine.mean()
    mse_loss = F.mse_loss(student.float(), teacher.float())
    return cosine_weight * cosine_loss + mse_weight * mse_loss, cosine_loss, mse_loss


def _batch_to_device(batch, device: str):
    """Move a vision tensor batch or a text {input_ids, attention_mask} dict batch."""
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


def _encode(model: nn.Module, batch, modality: str) -> torch.Tensor:
    """Dispatch to the modality's encoder. Text takes the int-input dict."""
    if modality == "vision":
        return model.encode_image(batch)
    return model.encode_text(batch)


@torch.no_grad()
def _evaluate(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: str,
    fake_quant: FakeQuantController,
    fake_quant_enabled: bool,
    modality: str = "vision",
) -> dict:
    student.eval()
    teacher.eval()
    context = nullcontext()
    if not fake_quant_enabled:
        context = fake_quant.disabled()

    cosines: list[float] = []
    mse_values: list[float] = []
    with context:
        for batch in loader:
            batch = _batch_to_device(batch, device)
            teacher_output = _encode(teacher, batch, modality).detach()
            student_output = _encode(student, batch, modality).detach()
            batch_cosine = _batch_cosine(student_output, teacher_output)
            cosines.extend(float(value) for value in batch_cosine.cpu())
            mse_values.append(
                float(F.mse_loss(student_output.float(), teacher_output.float()).cpu())
            )

    return {
        "num_samples": len(cosines),
        "cosine_l2_mean": sum(cosines) / len(cosines),
        "cosine_l2_min": min(cosines),
        "cosine_l2_max": max(cosines),
        "mse_mean": sum(mse_values) / len(mse_values),
    }


def _copy_export_metadata(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml",):
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Required export metadata not found: {source}")
        shutil.copy2(source, output_dir / name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_trainable_csv(path: Path, names: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter_name"])
        for name in names:
            writer.writerow([name])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vision-only quantization-robust mSigLIP student."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-input-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-input-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-list", default="input_list.txt")
    parser.add_argument("--precision", choices=["fp32"], default="fp32")
    parser.add_argument(
        "--modality",
        choices=["vision", "text"],
        default="vision",
        help=(
            "vision (default): distill encode_image on NCHW image .raw inputs. "
            "text: distill encode_text on int input_ids+attention_mask .raw inputs "
            "(SiglipTextTransformer, last-token Linear head). Same QAT machinery."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="Text only. Token sequence length of the .raw inputs. 0 = config tokenizer max length.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Optional global train-step cap.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="cosine: linear warmup (--warmup-frac) then cosine decay to "
        "--min-lr-ratio * lr. Classic QAT trick to squeeze a sharper minimum.",
    )
    parser.add_argument("--warmup-frac", type=float, default=0.0, help="Fraction of total steps for linear LR warmup.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0, help="Cosine floor as a fraction of --lr.")
    parser.add_argument("--mse-weight", type=float, default=0.05)
    parser.add_argument(
        "--clean-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for clean student-vs-teacher cosine consistency. Set 0 to "
            "recover the original fake-quant-only objective."
        ),
    )
    parser.add_argument(
        "--clean-mse-weight",
        type=float,
        default=0.05,
        help="MSE weight for the clean student-vs-teacher consistency path.",
    )
    parser.add_argument("--fake-quant-bits", type=int, default=8)
    parser.add_argument("--fake-quant-eps", type=float, default=1e-8)
    parser.add_argument(
        "--fake-quant-granularity",
        choices=["per_tensor", "per_sample"],
        default="per_tensor",
        help=(
            "per_tensor (default) matches AI Hub's deployed W8A8 (one scale per "
            "activation tensor) and transfers far better. per_sample is the legacy "
            "behavior (one scale per image): optimistic sim cosine, weak transfer."
        ),
    )
    parser.add_argument(
        "--fake-quant-observer",
        choices=["ema", "dynamic"],
        default="ema",
        help=(
            "ema (default): per-tensor scale from an EMA running max (fixed-ish, "
            "like AI Hub calibrate-once) -> best sim<->real transfer. dynamic: "
            "per-batch max each forward (granularity from --fake-quant-granularity)."
        ),
    )
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--start-layer", type=int, default=4)
    parser.add_argument("--end-layer", type=int, default=11)
    parser.add_argument("--no-train-visual-projection", action="store_true")
    parser.add_argument(
        "--quant-head",
        action="store_true",
        help=(
            "Also fake-quant + train the pooling head (post_layernorm, "
            "head.attention, head.mlp) — the last stage producing the embedding, "
            "whose INT8 error is not averaged out. Recommended to push R@1 higher."
        ),
    )
    parser.add_argument(
        "--quant-linears",
        action="store_true",
        help=(
            "Fake-quant every nn.Linear output (q/k/v/out_proj, fc1, fc2, head "
            "linears) in the selected layers, not just GELU + residual. Faithful "
            "coverage of AI Hub's per-activation quantization — the lever to break "
            "the per-tensor W8A8 plateau."
        ),
    )
    parser.add_argument(
        "--quant-attention",
        action="store_true",
        help=(
            "Also fake-quant the two intra-attention matmuls (Q*K^T scores, softmax "
            "probs, probs*V context) by patching the eager attention forward of the "
            "selected encoder layers. Covers the activations --quant-linears cannot "
            "reach (functional matmuls). The last coverage lever toward R@1 >= 50."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    print("Loading teacher/student models")
    print(f"  model_dir: {model_dir}")
    print(f"  device:    {device}")
    print(f"  modality:  {args.modality}")
    teacher, config = _load_pytorch_model(model_dir, args.precision, device)
    student = copy.deepcopy(teacher)

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    seq_len = 0
    if args.modality == "text":
        seq_len = args.seq_len or _text_seq_len(config)
        print(f"  seq_len:   {seq_len}")

    trainable_names = _freeze_for_qat(
        student,
        modality=args.modality,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        train_projection=not args.no_train_visual_projection,
        train_head=args.quant_head,
    )
    trainable_count, total_count = _trainable_parameter_count(student)
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    fake_quant = FakeQuantController(
        student,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        bits=args.fake_quant_bits,
        eps=args.fake_quant_eps,
        per_tensor=args.fake_quant_granularity == "per_tensor",
        observer=args.fake_quant_observer,
        ema_momentum=args.ema_momentum,
        quant_head=args.quant_head,
        quant_linears=args.quant_linears,
        quant_attention=args.quant_attention,
        modality=args.modality,
    )
    fake_quant.install()
    print("Fake-quant hooks:")
    for name in fake_quant.hooked_modules:
        print(f"  - {name}")

    if args.modality == "vision":
        train_dataset: Dataset = RawVisionDataset(
            args.train_input_dir, args.input_list, args.image_size, max_samples=args.max_train_samples
        )
        val_dataset: Dataset = RawVisionDataset(
            args.val_input_dir, args.input_list, args.image_size, max_samples=args.max_val_samples
        )
    else:
        train_dataset = RawTextDataset(
            args.train_input_dir, args.input_list, seq_len, max_samples=args.max_train_samples
        )
        val_dataset = RawTextDataset(
            args.val_input_dir, args.input_list, seq_len, max_samples=args.max_val_samples
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule: optional linear warmup then cosine decay to min_lr_ratio*lr.
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    total_steps = max(1, total_steps)
    warmup_steps = int(args.warmup_frac * total_steps)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if args.lr_schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    print(
        f"LR schedule: {args.lr_schedule} | total_steps={total_steps} "
        f"warmup={warmup_steps} min_lr_ratio={args.min_lr_ratio}"
    )

    history: list[dict] = []
    global_step = 0
    stop_training = False
    for epoch in range(args.epochs):
        student.train()
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_output = _encode(teacher, batch, args.modality).detach()

            with fake_quant.disabled():
                clean_student_output = _encode(student, batch, args.modality)
            fake_student_output = _encode(student, batch, args.modality)

            clean_loss, clean_cosine_loss, clean_mse_loss = _weighted_distillation_loss(
                clean_student_output,
                teacher_output,
                cosine_weight=args.clean_weight,
                mse_weight=args.clean_mse_weight,
            )
            fake_loss, fake_cosine_loss, fake_mse_loss = _distillation_loss(
                fake_student_output,
                teacher_output,
                mse_weight=args.mse_weight,
            )
            loss = fake_loss + clean_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            cur_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()

            clean_cos_mean = (
                _batch_cosine(clean_student_output.detach(), teacher_output).mean().item()
            )
            fake_cos_mean = (
                _batch_cosine(fake_student_output.detach(), teacher_output).mean().item()
            )
            row = {
                "epoch": epoch,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "fake_loss": float(fake_loss.detach().cpu()),
                "fake_cosine_loss": float(fake_cosine_loss.detach().cpu()),
                "fake_mse_loss": float(fake_mse_loss.detach().cpu()),
                "fake_train_cosine_l2_mean": fake_cos_mean,
                "clean_loss": float(clean_loss.detach().cpu()),
                "clean_cosine_loss": float(clean_cosine_loss.detach().cpu()),
                "clean_mse_loss": float(clean_mse_loss.detach().cpu()),
                "clean_train_cosine_l2_mean": clean_cos_mean,
                "lr": cur_lr,
                # Keep the legacy key for quick comparisons with earlier summaries.
                "train_cosine_l2_mean": fake_cos_mean,
            }
            history.append(row)
            print(
                "step={step} epoch={epoch} lr={lr:.2e} loss={loss:.6f} "
                "fake_cos={fake_train_cosine_l2_mean:.6f} "
                "fake_mse={fake_mse_loss:.6f} "
                "clean_cos={clean_train_cosine_l2_mean:.6f} "
                "clean_mse={clean_mse_loss:.6f}".format(**row)
            )

            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    clean_eval = _evaluate(
        teacher, student, val_loader, device, fake_quant, fake_quant_enabled=False, modality=args.modality
    )
    fake_eval = _evaluate(
        teacher, student, val_loader, device, fake_quant, fake_quant_enabled=True, modality=args.modality
    )

    fake_quant.close()
    student.eval()
    student = student.cpu()

    _copy_export_metadata(model_dir, output_dir)
    torch.save(student.state_dict(), output_dir / "model_fp32.pt")

    summary = {
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "modality": args.modality,
        "seq_len": seq_len if args.modality == "text" else None,
        "train_input_dir": str(args.train_input_dir.expanduser().resolve()),
        "val_input_dir": str(args.val_input_dir.expanduser().resolve()),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr_schedule": args.lr_schedule,
        "warmup_frac": args.warmup_frac,
        "min_lr_ratio": args.min_lr_ratio,
        "global_steps": global_step,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mse_weight": args.mse_weight,
        "clean_weight": args.clean_weight,
        "clean_mse_weight": args.clean_mse_weight,
        "fake_quant_bits": args.fake_quant_bits,
        "fake_quant_granularity": args.fake_quant_granularity,
        "fake_quant_observer": args.fake_quant_observer,
        "ema_momentum": args.ema_momentum,
        "quant_head": args.quant_head,
        "quant_linears": args.quant_linears,
        "quant_attention": args.quant_attention,
        "layer_range": [args.start_layer, args.end_layer],
        "train_visual_projection": not args.no_train_visual_projection,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "val_clean": clean_eval,
        "val_fake_quant": fake_eval,
        "history": history,
    }
    summary_name = "vision_quant_robust_summary.json" if args.modality == "vision" else "text_quant_robust_summary.json"
    _write_json(output_dir / summary_name, summary)
    _write_trainable_csv(output_dir / "trainable_parameters.csv", trainable_names)

    print("\nSaved QAT/fine-tuned FP32 export:")
    print(f"  {output_dir}")
    print("Validation:")
    print(json.dumps({"clean": clean_eval, "fake_quant": fake_eval}, indent=2))
    print("\nNext export command:")
    if args.modality == "vision":
        print(
            "  python3 deployment/scripts/qnn/export_rotated_vision_onnx.py "
            f"--model-dir {_display_path(output_dir)} --opset 20"
        )
    else:
        print(
            "  python3 deployment/scripts/qnn/export_text_onnx.py "
            f"--model-dir {_display_path(output_dir)}"
        )


if __name__ == "__main__":
    main()

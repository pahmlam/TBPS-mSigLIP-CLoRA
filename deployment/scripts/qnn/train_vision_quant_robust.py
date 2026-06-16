#!/usr/bin/env python3
"""Vision-only quantization-aware fine-tuning by teacher-student distillation.

This script does not export a custom QDQ graph. It fine-tunes the FP32 vision
encoder while injecting lightweight fake-quant noise into the sensitive vision
blocks, then saves a normal FP32 deployment export directory. The follow-up
step is to run the existing ONNX export and AI Hub native quantizer on that
tuned FP32 model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import sys
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


class FakeQuantController:
    """Installs activation fake-quant hooks for selected SigLIP vision blocks."""

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
    ) -> None:
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
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

    def install(self) -> None:
        layers = self.model.backbone.vision_model.encoder.layers
        if self.start_layer < 0 or self.end_layer >= len(layers):
            raise ValueError(
                f"Layer range {self.start_layer}-{self.end_layer} is invalid for "
                f"{len(layers)} vision encoder layers."
            )

        for index in range(self.start_layer, self.end_layer + 1):
            layer = layers[index]
            self._hook_module(
                f"backbone.vision_model.encoder.layers.{index}.mlp.activation_fn",
                layer.mlp.activation_fn,
                output_is_tuple=False,
            )
            self._hook_module(
                f"backbone.vision_model.encoder.layers.{index}",
                layer,
                output_is_tuple=True,
            )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _freeze_for_vision_qat(
    model: nn.Module,
    start_layer: int,
    end_layer: int,
    train_visual_projection: bool,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_prefixes = [
        f"backbone.vision_model.encoder.layers.{index}."
        for index in range(start_layer, end_layer + 1)
    ]
    if train_visual_projection and hasattr(model.backbone, "visual_projection"):
        trainable_prefixes.append("backbone.visual_projection.")

    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError("No trainable parameters selected for vision QAT.")
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


@torch.no_grad()
def _evaluate(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: str,
    fake_quant: FakeQuantController,
    fake_quant_enabled: bool,
) -> dict:
    student.eval()
    teacher.eval()
    context = nullcontext()
    if not fake_quant_enabled:
        context = fake_quant.disabled()

    cosines: list[float] = []
    mse_values: list[float] = []
    with context:
        for images in loader:
            images = images.to(device, non_blocking=True)
            teacher_output = teacher.encode_image(images).detach()
            student_output = student.encode_image(images).detach()
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Optional global train-step cap.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    teacher, _ = _load_pytorch_model(model_dir, args.precision, device)
    student = copy.deepcopy(teacher)

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    trainable_names = _freeze_for_vision_qat(
        student,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        train_visual_projection=not args.no_train_visual_projection,
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
    )
    fake_quant.install()
    print("Fake-quant hooks:")
    for name in fake_quant.hooked_modules:
        print(f"  - {name}")

    train_dataset = RawVisionDataset(
        args.train_input_dir,
        args.input_list,
        args.image_size,
        max_samples=args.max_train_samples,
    )
    val_dataset = RawVisionDataset(
        args.val_input_dir,
        args.input_list,
        args.image_size,
        max_samples=args.max_val_samples,
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

    history: list[dict] = []
    global_step = 0
    stop_training = False
    for epoch in range(args.epochs):
        student.train()
        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_output = teacher.encode_image(images).detach()

            with fake_quant.disabled():
                clean_student_output = student.encode_image(images)
            fake_student_output = student.encode_image(images)

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
            optimizer.step()

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
                # Keep the legacy key for quick comparisons with earlier summaries.
                "train_cosine_l2_mean": fake_cos_mean,
            }
            history.append(row)
            print(
                "step={step} epoch={epoch} loss={loss:.6f} "
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
        teacher, student, val_loader, device, fake_quant, fake_quant_enabled=False
    )
    fake_eval = _evaluate(
        teacher, student, val_loader, device, fake_quant, fake_quant_enabled=True
    )

    fake_quant.close()
    student.eval()
    student = student.cpu()

    _copy_export_metadata(model_dir, output_dir)
    torch.save(student.state_dict(), output_dir / "model_fp32.pt")

    summary = {
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "train_input_dir": str(args.train_input_dir.expanduser().resolve()),
        "val_input_dir": str(args.val_input_dir.expanduser().resolve()),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
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
    _write_json(output_dir / "vision_quant_robust_summary.json", summary)
    _write_trainable_csv(output_dir / "trainable_parameters.csv", trainable_names)

    print("\nSaved QAT/fine-tuned FP32 export:")
    print(f"  {output_dir}")
    print("Validation:")
    print(json.dumps({"clean": clean_eval, "fake_quant": fake_eval}, indent=2))
    print("\nNext export command:")
    print(
        "  python3 deployment/scripts/onnx/export.py "
        f"--model-dir {_display_path(output_dir)} --precision fp32"
    )


if __name__ == "__main__":
    main()

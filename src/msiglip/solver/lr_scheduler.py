import warnings
from typing import Optional, List
from bisect import bisect_right
from math import cos, pi
from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        mode: str,
        warmup_epochs: int = 10,
        warmup_method: str = "linear",
        total_epochs: int = 100,
        n_iter_per_epoch: int = 100,
        last_epoch: int = -1,
        milestones: Optional[List[int]] = None,
        gamma: Optional[float] = None,
        warmup_factor: Optional[float] = None,
        power: Optional[float] = None,
    ):
        """
        LRSchedulerWithWarmup with iteration-based learning rate scheduling.
        Handles multiple parameter groups with different base learning rates.
        Each parameter group should contain 'lr', 'start_lr', and 'end_lr'.

        Args:
            optimizer: Wrapped optimizer
            mode: Scheduling mode ('step', 'exp', 'poly', 'cosine', 'linear', 'constant')
            warmup_epochs: Number of epochs for warmup phase
            warmup_method: Warmup method ('constant', 'linear' or None)
            total_epochs: Total number of epochs
            n_iter_per_epoch: Number of iterations per epoch
            last_epoch: The index of last epoch, -1 for start
            milestones: List of epoch numbers for step mode
            gamma: Multiplicative factor for step mode
            warmup_factor: Factor for warmup scaling
        """
        # Basic parameter validation
        warmup_epochs = int(warmup_epochs)
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer")

        if not isinstance(total_epochs, int) or total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer")

        if warmup_epochs >= total_epochs:
            raise ValueError("warmup_epochs must be less than total_epochs")

        if mode not in ("step", "exp", "poly", "cosine", "linear", "constant"):
            raise ValueError(
                f"Invalid mode {mode}. Choose from 'step', 'exp', 'poly', 'cosine', 'linear', 'constant'"
            )

        if warmup_method not in ("constant", "linear", None):
            raise ValueError(
                f"Invalid warmup_method {warmup_method}. Choose 'constant' or 'linear' or None"
            )

        # Validate param groups have required learning rates
        for i, group in enumerate(optimizer.param_groups):
            required_params = ["lr", "start_lr", "end_lr"]
            missing_params = [p for p in required_params if p not in group]
            if missing_params:
                raise ValueError(
                    f"Parameter group {i} is missing required parameters: {missing_params}"
                )
            if not isinstance(group["start_lr"], (int, float)) or group["start_lr"] < 0:
                raise ValueError(f"start_lr must be non-negative in group {i}")
            if not isinstance(group["end_lr"], (int, float)) or group["end_lr"] < 0:
                raise ValueError(f"end_lr must be non-negative in group {i}")
            if group["start_lr"] < group["end_lr"]:
                warnings.warn(
                    f"start_lr is less than end_lr in group {i}, learning rate will increase over time"
                )

        # Get all learning rates from param groups
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.start_lrs = [group["start_lr"] for group in optimizer.param_groups]
        self.end_lrs = [group["end_lr"] for group in optimizer.param_groups]

        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.n_iter_per_epoch = n_iter_per_epoch

        # Convert epochs to iterations
        self.total_iters = total_epochs * n_iter_per_epoch
        self.warmup_iters = warmup_epochs * n_iter_per_epoch

        # Mode-specific parameter validation
        if mode == "step":
            if not isinstance(milestones, (list, tuple)) or not milestones:
                raise ValueError("milestones must be a non-empty list or tuple")
            if not isinstance(gamma, (int, float)) or gamma <= 0:
                raise ValueError("gamma must be a positive number")
            if not all(isinstance(m, int) and m > 0 for m in milestones):
                raise ValueError("all milestones must be positive integers")
            if not list(milestones) == sorted(milestones):
                raise ValueError(
                    f"Milestones should be in ascending order. Got {milestones}"
                )
            if max(milestones) >= total_epochs:
                raise ValueError("all milestones should be less than total_epochs")

            self.milestones = [m * n_iter_per_epoch for m in milestones]
            self.gamma = gamma

        elif mode == "poly":
            if not isinstance(power, (int, float)) or power <= 0:
                raise ValueError("power must be a positive number for poly mode")
            self.power = power

        # Warmup validation and defaults
        if warmup_method == "constant":
            self.warmup_factor = 1.0 if warmup_factor is None else warmup_factor
        elif warmup_method == "linear":
            self.warmup_factor = 0.0 if warmup_factor is None else warmup_factor

        if warmup_factor is not None and not isinstance(warmup_factor, (int, float)):
            raise ValueError("warmup_factor must be a number")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rates for current iteration.
        Maintains relative scaling between parameter groups throughout training.

        Returns:
            List of learning rates for each parameter group
        """
        current_iter = self.last_epoch

        # Warmup phase
        if self.warmup_method and current_iter < self.warmup_iters:
            if self.warmup_method == "constant":
                return [base_lr * self.warmup_factor for base_lr in self.base_lrs]
            elif self.warmup_method == "linear":
                alpha = current_iter / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                return [
                    start_lr + (base_lr - start_lr) * warmup_factor
                    for start_lr, base_lr in zip(self.start_lrs, self.base_lrs)
                ]

        # Post-warmup phase
        iter_ratio = (current_iter - self.warmup_iters) / (
            self.total_iters - self.warmup_iters
        )

        if self.mode == "step":
            decay = self.gamma ** bisect_right(self.milestones, current_iter)
            return [base_lr * decay for base_lr in self.base_lrs]
        elif self.mode in ("exp", "linear"):
            return [base_lr * (1 - iter_ratio) for base_lr in self.base_lrs]
        elif self.mode == "poly":
            factor = (1 - iter_ratio) ** self.power
            return [
                end_lr + (base_lr - end_lr) * factor
                for base_lr, end_lr in zip(self.base_lrs, self.end_lrs)
            ]
        elif self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * iter_ratio))
            return [
                end_lr + (base_lr - end_lr) * factor
                for base_lr, end_lr in zip(self.base_lrs, self.end_lrs)
            ]
        elif self.mode == "constant":
            return self.base_lrs

        raise ValueError(f"Invalid mode {self.mode}")

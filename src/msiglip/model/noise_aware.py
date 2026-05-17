"""
Noise-Aware Circle Loss state management (Idea C).

This module provides `NoiseAwareCircleState`, an `nn.Module` that owns:
  - Running EMA of positive/negative similarity distributions (for Bayesian FN detection)
  - Per-sample EMA of Circle Loss values (for GMM-based FP detection)
  - 2-component 1D GMM fit (pure PyTorch, no sklearn dependency)

The loss function in `model/objectives.py:compute_noise_aware_circle()` remains
stateless — this module merely provides the statistics it consumes.

References:
  - knowledge/noise_ideas_math.md — Section "Idea C"
  - knowledge/noise_ideas_concepts.md — Section "Unified Noise-Aware Circle Loss"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from loguru import logger


class NoiseAwareCircleState(nn.Module):
    """
    Stateful companion to `compute_noise_aware_circle()`.

    All running statistics are `register_buffer`'d so they survive checkpointing
    and move with `.to(device)`.

    Args:
        num_train_samples: Size of the training dataset. Sized once at construction
            time; each batch index (`batch["id"]`) must be < num_train_samples.
        config: OmegaConf / dict with noise-detection hyperparameters. Missing keys
            fall back to the defaults below.
    """

    def __init__(self, num_train_samples: int, config):
        super().__init__()

        if num_train_samples <= 0:
            raise ValueError(
                f"num_train_samples must be positive, got {num_train_samples}"
            )

        # -------- Hyperparameters --------
        def _get(key, default):
            if hasattr(config, "get"):
                return config.get(key, default)
            return getattr(config, key, default) if config is not None else default

        self.ema_beta: float = float(_get("ema_beta", 0.99))
        self.loss_ema_alpha: float = float(_get("loss_ema_alpha", 0.9))
        self.fn_prior: float = float(_get("fn_prior", 0.01))
        self.epsilon_n: float = float(_get("epsilon_n", 0.1))
        self.epsilon_p: float = float(_get("epsilon_p", 0.2))
        self.gmm_refit_interval: int = int(_get("gmm_refit_interval", 5))
        self.gmm_min_separation: float = float(_get("gmm_min_separation", 1.0))
        self.fn_enable_epoch: int = int(_get("fn_enable_epoch", 11))
        self.fp_enable_epoch: int = int(_get("fp_enable_epoch", 15))

        # -------- FN detection buffers (EMA similarity stats) --------
        # Initial values are placeholders; overwritten on first update.
        self.register_buffer("mu_pos", torch.tensor(0.5))
        self.register_buffer("sigma_pos", torch.tensor(0.2))
        self.register_buffer("mu_neg", torch.tensor(-0.1))
        self.register_buffer("sigma_neg", torch.tensor(0.2))
        # 0.0 = not yet initialized, 1.0 = initialized. Use float for buffer compatibility.
        self.register_buffer("stats_initialized", torch.tensor(0.0))

        # -------- FP detection buffers (per-sample loss EMA + clean weights) --------
        self.num_train_samples = int(num_train_samples)
        self.register_buffer("ema_loss", torch.zeros(self.num_train_samples))
        self.register_buffer(
            "sample_seen", torch.zeros(self.num_train_samples, dtype=torch.bool)
        )
        # Clean probability per sample; initialize to 1.0 (= treat as clean) so the
        # loss is unaffected before the GMM is fit.
        self.register_buffer("clean_weights", torch.ones(self.num_train_samples))
        self.register_buffer("gmm_fitted", torch.tensor(0.0))

    # -------------------------------------------------------------------------
    # EMA similarity statistics (for FN detection)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_ema_stats(self, s_p: torch.Tensor, s_n: torch.Tensor) -> None:
        """
        Update running EMA of similarity means/stds.

        On first call, initializes directly from batch stats (no momentum lag).
        On subsequent calls, EMA with `self.ema_beta`.

        Args:
            s_p: (Np,) flattened positive similarities
            s_n: (Nn,) flattened negative similarities
        """
        if s_p.numel() < 2 or s_n.numel() < 2:
            # Need at least 2 samples to compute std; skip this step.
            return

        # Detach in case caller didn't.
        s_p = s_p.detach().float()
        s_n = s_n.detach().float()

        batch_mu_pos = s_p.mean()
        batch_sigma_pos = s_p.std(unbiased=False).clamp_min(1e-4)
        batch_mu_neg = s_n.mean()
        batch_sigma_neg = s_n.std(unbiased=False).clamp_min(1e-4)

        if self.stats_initialized.item() < 0.5:
            # First call: initialize directly.
            self.mu_pos.copy_(batch_mu_pos)
            self.sigma_pos.copy_(batch_sigma_pos)
            self.mu_neg.copy_(batch_mu_neg)
            self.sigma_neg.copy_(batch_sigma_neg)
            self.stats_initialized.fill_(1.0)
            return

        beta = self.ema_beta
        self.mu_pos.mul_(beta).add_(batch_mu_pos, alpha=1.0 - beta)
        self.sigma_pos.mul_(beta).add_(batch_sigma_pos, alpha=1.0 - beta)
        self.mu_neg.mul_(beta).add_(batch_mu_neg, alpha=1.0 - beta)
        self.sigma_neg.mul_(beta).add_(batch_sigma_neg, alpha=1.0 - beta)

    def get_fn_stats_dict(self) -> dict:
        """
        Snapshot of EMA stats as Python floats — the format consumed by
        `compute_noise_aware_circle(fn_stats=...)`.

        Returns None if stats have not yet been initialized.
        """
        if self.stats_initialized.item() < 0.5:
            return None
        return {
            "mu_pos": self.mu_pos.item(),
            "sigma_pos": self.sigma_pos.item(),
            "mu_neg": self.mu_neg.item(),
            "sigma_neg": self.sigma_neg.item(),
            "fn_prior": self.fn_prior,
        }

    # -------------------------------------------------------------------------
    # Per-sample loss tracking (for FP detection)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_sample_losses(
        self, sample_ids: torch.Tensor, per_sample_losses: torch.Tensor
    ) -> None:
        """
        Update per-sample EMA loss buffer.

        For unseen samples, initialize directly (no EMA lag).
        For seen samples, apply `ema = alpha * ema + (1-alpha) * current`.

        If `sample_ids` contains duplicates (can happen with RandomIdentitySampler
        when a PID has fewer instances than `num_instances`), the **last write
        wins** — this is practically equivalent to EMA with a slightly smaller
        effective alpha for that step and avoids a costly segment_reduce.

        Args:
            sample_ids: (B,) long tensor of dataset indices
            per_sample_losses: (B,) float tensor of this-step losses per sample
        """
        sample_ids = sample_ids.detach().long().view(-1)
        per_sample_losses = per_sample_losses.detach().float().view(-1)

        if sample_ids.numel() == 0:
            return

        # Move to the same device as our buffers if needed.
        device = self.ema_loss.device
        sample_ids = sample_ids.to(device)
        per_sample_losses = per_sample_losses.to(device)

        # Bounds check — guards against off-by-one if dataset size changes.
        valid = (sample_ids >= 0) & (sample_ids < self.num_train_samples)
        if not valid.all():
            sample_ids = sample_ids[valid]
            per_sample_losses = per_sample_losses[valid]
            if sample_ids.numel() == 0:
                return

        seen_mask = self.sample_seen[sample_ids]
        unseen_mask = ~seen_mask

        # Unseen samples: direct assignment (no EMA lag).
        if unseen_mask.any():
            unseen_ids = sample_ids[unseen_mask]
            self.ema_loss[unseen_ids] = per_sample_losses[unseen_mask]
            self.sample_seen[unseen_ids] = True

        # Seen samples: EMA update.
        if seen_mask.any():
            seen_ids = sample_ids[seen_mask]
            alpha = self.loss_ema_alpha
            current = self.ema_loss[seen_ids]
            self.ema_loss[seen_ids] = alpha * current + (1.0 - alpha) * per_sample_losses[seen_mask]

    @torch.no_grad()
    def get_clean_weights_for_batch(self, sample_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up clean-probability weights for a batch of dataset indices.

        If the GMM has not been fit yet, returns all-ones (no suppression).
        """
        sample_ids = sample_ids.detach().long().view(-1).to(self.clean_weights.device)
        if self.gmm_fitted.item() < 0.5:
            return torch.ones_like(sample_ids, dtype=self.clean_weights.dtype)
        return self.clean_weights[sample_ids]

    # -------------------------------------------------------------------------
    # GMM fitting (FP detection core)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def refit_gmm(self) -> dict:
        """
        Fit a 2-component 1D GMM to the per-sample EMA losses and update
        `self.clean_weights` accordingly.

        Returns a diagnostics dict with keys:
          - separation: |mu_clean - mu_noisy| / (sigma_clean + sigma_noisy)
          - mu_clean, mu_noisy, sigma_clean, sigma_noisy
          - pi_clean, pi_noisy
          - fallback: 1.0 if separation was below threshold and we reverted to
            uniform weights, 0.0 otherwise
          - n_samples_used: how many samples the GMM was fit on
        """
        seen_idx = self.sample_seen.nonzero(as_tuple=False).squeeze(-1)
        if seen_idx.numel() < 50:
            # Not enough data to fit a meaningful 2-component GMM.
            logger.warning(
                f"GMM refit skipped: only {seen_idx.numel()} samples seen so far (need >= 50)."
            )
            return {
                "separation": 0.0,
                "mu_clean": 0.0,
                "mu_noisy": 0.0,
                "sigma_clean": 0.0,
                "sigma_noisy": 0.0,
                "pi_clean": 1.0,
                "pi_noisy": 0.0,
                "fallback": 1.0,
                "n_samples_used": float(seen_idx.numel()),
            }

        data = self.ema_loss[seen_idx]
        mu_0, sigma_0, pi_0, mu_1, sigma_1, pi_1 = self._fit_1d_gmm_em(data)

        # Identify the clean component (lower mean = less loss = cleaner sample).
        if mu_0.item() <= mu_1.item():
            mu_c, sigma_c, pi_c = mu_0, sigma_0, pi_0
            mu_nz, sigma_nz, pi_nz = mu_1, sigma_1, pi_1
        else:
            mu_c, sigma_c, pi_c = mu_1, sigma_1, pi_1
            mu_nz, sigma_nz, pi_nz = mu_0, sigma_0, pi_0

        separation = (mu_nz - mu_c).abs() / (sigma_c + sigma_nz + 1e-8)

        # If distributions are indistinguishable, fall back to uniform weights.
        fallback = separation.item() < self.gmm_min_separation
        if fallback:
            logger.warning(
                f"GMM separation {separation.item():.3f} < threshold "
                f"{self.gmm_min_separation} — falling back to uniform clean_weights."
            )
            self.clean_weights.fill_(1.0)
        else:
            # Posterior P(clean | loss_i) for each seen sample.
            log_p_c = self._log_gaussian(data, mu_c, sigma_c) + torch.log(pi_c + 1e-12)
            log_p_n = self._log_gaussian(data, mu_nz, sigma_nz) + torch.log(pi_nz + 1e-12)
            # Numerically stable log-sum-exp for the denominator.
            log_denom = torch.logaddexp(log_p_c, log_p_n)
            clean_prob = (log_p_c - log_denom).exp()

            # Reset to 1.0 (clean-by-default), then write updated probs for seen samples.
            self.clean_weights.fill_(1.0)
            self.clean_weights[seen_idx] = clean_prob.to(self.clean_weights.dtype)

        self.gmm_fitted.fill_(1.0)

        return {
            "separation": separation.item(),
            "mu_clean": mu_c.item(),
            "mu_noisy": mu_nz.item(),
            "sigma_clean": sigma_c.item(),
            "sigma_noisy": sigma_nz.item(),
            "pi_clean": pi_c.item(),
            "pi_noisy": pi_nz.item(),
            "fallback": 1.0 if fallback else 0.0,
            "n_samples_used": float(seen_idx.numel()),
        }

    @staticmethod
    def _log_gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Log-density of N(mu, sigma^2) at x (all 1D)."""
        sigma = sigma.clamp_min(1e-6)
        return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)

    @staticmethod
    def _fit_1d_gmm_em(
        data: torch.Tensor, n_iter: int = 50, tol: float = 1e-6
    ) -> tuple:
        """
        Fit 2-component 1D GMM via Expectation-Maximization.

        Returns:
            (mu_0, sigma_0, pi_0, mu_1, sigma_1, pi_1) — each a 0-dim tensor.
        """
        data = data.detach().float().view(-1)
        n = data.numel()

        # Initialization: two quantile points of the data (robust to outliers).
        q25, q75 = torch.quantile(data, torch.tensor([0.25, 0.75], device=data.device))
        mu_0 = q25.clone()
        mu_1 = q75.clone()
        sigma_0 = data.std(unbiased=False).clamp_min(1e-4)
        sigma_1 = sigma_0.clone()
        pi_0 = torch.tensor(0.5, device=data.device)
        pi_1 = torch.tensor(0.5, device=data.device)

        prev_log_lik = torch.tensor(-float("inf"), device=data.device)

        for _ in range(n_iter):
            # E-step: compute responsibilities in log-space for stability.
            log_p_0 = NoiseAwareCircleState._log_gaussian(data, mu_0, sigma_0) + torch.log(pi_0 + 1e-12)
            log_p_1 = NoiseAwareCircleState._log_gaussian(data, mu_1, sigma_1) + torch.log(pi_1 + 1e-12)
            log_denom = torch.logaddexp(log_p_0, log_p_1)
            r_0 = (log_p_0 - log_denom).exp()
            r_1 = 1.0 - r_0

            # M-step: update parameters.
            n_0 = r_0.sum().clamp_min(1e-6)
            n_1 = r_1.sum().clamp_min(1e-6)

            mu_0_new = (r_0 * data).sum() / n_0
            mu_1_new = (r_1 * data).sum() / n_1
            sigma_0_new = ((r_0 * (data - mu_0_new) ** 2).sum() / n_0).sqrt().clamp_min(1e-4)
            sigma_1_new = ((r_1 * (data - mu_1_new) ** 2).sum() / n_1).sqrt().clamp_min(1e-4)
            pi_0_new = n_0 / n
            pi_1_new = n_1 / n

            mu_0, mu_1 = mu_0_new, mu_1_new
            sigma_0, sigma_1 = sigma_0_new, sigma_1_new
            pi_0, pi_1 = pi_0_new, pi_1_new

            # Check log-likelihood convergence.
            log_lik = log_denom.sum()
            if (log_lik - prev_log_lik).abs() < tol:
                break
            prev_log_lik = log_lik

        return mu_0, sigma_0, pi_0, mu_1, sigma_1, pi_1

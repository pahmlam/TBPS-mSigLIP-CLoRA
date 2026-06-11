"""
Evidence memory for MNEB-HN.

The bank is intentionally separate from Circle Loss. It collects cross-epoch
evidence for FN/FP auxiliary losses while preserving the vanilla hard-negative
path unless MNEB auxiliary losses are explicitly enabled.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceMemoryBank(nn.Module):
    """
    Cross-epoch state for MNEB-HN.

    Buffer values move with the model and survive checkpoints. Runtime-only
    sparse top-k counts are kept in a plain dict because they are diagnostics,
    not critical training state.
    """

    def __init__(self, num_train_samples: int, embed_dim: int, config):
        super().__init__()
        if num_train_samples <= 0:
            raise ValueError(
                f"num_train_samples must be positive for MNEB, got {num_train_samples}"
            )

        def _get(cfg, key, default):
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        evidence_cfg = _get(config, "evidence_bank", {})
        fnm_cfg = _get(config, "fnm_aux", {})
        rde_cfg = _get(config, "rde_aux", {})

        self.num_train_samples = int(num_train_samples)
        self.embed_dim = int(embed_dim)
        self.queue_size = int(_get(evidence_cfg, "queue_size", 4096))
        self.topk = int(_get(evidence_cfg, "topk", 20))
        self.ema_beta = float(_get(evidence_cfg, "ema_beta", 0.99))
        self.loss_ema_alpha = float(_get(evidence_cfg, "loss_ema_alpha", 0.9))
        self.min_seen_epochs = int(_get(evidence_cfg, "min_seen_epochs", 3))
        self.min_gmm_samples = int(_get(evidence_cfg, "min_gmm_samples", 50))
        self.gmm_interval = int(_get(evidence_cfg, "gmm_interval", 5))
        self.min_gmm_separation = float(_get(evidence_cfg, "min_gmm_separation", 1.0))
        self.clean_threshold = float(_get(evidence_cfg, "clean_threshold", 0.5))

        self.fn_prior = float(_get(fnm_cfg, "fn_prior", 0.003))
        self.fnm_enable_epoch = int(_get(fnm_cfg, "enable_epoch", 15))
        self.rde_enable_epoch = int(_get(rde_cfg, "enable_epoch", 15))

        self.register_buffer("image_global", torch.zeros(self.num_train_samples, self.embed_dim))
        self.register_buffer("text_global", torch.zeros(self.num_train_samples, self.embed_dim))
        self.register_buffer("image_local", torch.zeros(self.num_train_samples, self.embed_dim))
        self.register_buffer("text_local", torch.zeros(self.num_train_samples, self.embed_dim))
        self.register_buffer("global_loss_ema", torch.zeros(self.num_train_samples))
        self.register_buffer("local_loss_ema", torch.zeros(self.num_train_samples))
        self.register_buffer("sample_seen", torch.zeros(self.num_train_samples, dtype=torch.bool))
        self.register_buffer("local_seen", torch.zeros(self.num_train_samples, dtype=torch.bool))
        self.register_buffer("seen_counts", torch.zeros(self.num_train_samples, dtype=torch.long))

        self.register_buffer("global_clean_prob", torch.ones(self.num_train_samples))
        self.register_buffer("local_clean_prob", torch.ones(self.num_train_samples))
        self.register_buffer(
            "consensus_label",
            torch.full((self.num_train_samples,), -1, dtype=torch.long),
        )

        self.register_buffer("queue_ids", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("queue_count", torch.tensor(0, dtype=torch.long))

        self.register_buffer("mu_pos", torch.tensor(0.5))
        self.register_buffer("sigma_pos", torch.tensor(0.2))
        self.register_buffer("mu_neg", torch.tensor(-0.1))
        self.register_buffer("sigma_neg", torch.tensor(0.2))
        self.register_buffer("fn_stats_initialized", torch.tensor(0.0))

        self.register_buffer("global_gmm_fitted", torch.tensor(0.0))
        self.register_buffer("local_gmm_fitted", torch.tensor(0.0))
        self.register_buffer("global_gmm_fallback", torch.tensor(1.0))
        self.register_buffer("local_gmm_fallback", torch.tensor(1.0))
        self.register_buffer("last_refit_epoch", torch.tensor(-1, dtype=torch.long))

        self.topk_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    @torch.no_grad()
    def update_batch(
        self,
        sample_ids: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pids: torch.Tensor,
        global_losses: torch.Tensor | None = None,
        local_losses: torch.Tensor | None = None,
        local_image_features: torch.Tensor | None = None,
        local_text_features: torch.Tensor | None = None,
    ) -> dict:
        """Update per-sample memory and in-batch distribution statistics."""
        sample_ids = sample_ids.detach().long().view(-1).to(self.sample_seen.device)
        valid = (sample_ids >= 0) & (sample_ids < self.num_train_samples)
        if not valid.all():
            sample_ids = sample_ids[valid]
            image_features = image_features[valid]
            text_features = text_features[valid]
            pids = pids[valid]
            if global_losses is not None:
                global_losses = global_losses[valid]
            if local_losses is not None:
                local_losses = local_losses[valid]
            if local_image_features is not None:
                local_image_features = local_image_features[valid]
            if local_text_features is not None:
                local_text_features = local_text_features[valid]
        if sample_ids.numel() == 0:
            return self.diagnostics()

        image_features = F.normalize(image_features.detach(), dim=1, p=2).to(self.image_global.device)
        text_features = F.normalize(text_features.detach(), dim=1, p=2).to(self.text_global.device)
        pids = pids.detach().to(image_features.device)

        previously_seen = self.sample_seen[sample_ids].clone()
        self._update_feature_buffer(self.image_global, sample_ids, image_features, previously_seen)
        self._update_feature_buffer(self.text_global, sample_ids, text_features, previously_seen)

        if local_image_features is not None and local_text_features is not None:
            local_image_features = F.normalize(local_image_features.detach(), dim=1, p=2).to(self.image_local.device)
            local_text_features = F.normalize(local_text_features.detach(), dim=1, p=2).to(self.text_local.device)
            previously_local_seen = self.local_seen[sample_ids].clone()
            self._update_feature_buffer(self.image_local, sample_ids, local_image_features, previously_local_seen)
            self._update_feature_buffer(self.text_local, sample_ids, local_text_features, previously_local_seen)
            self.local_seen[sample_ids] = True

        if global_losses is not None:
            self._update_loss_buffer(self.global_loss_ema, sample_ids, global_losses, previously_seen)
        if local_losses is not None:
            self._update_loss_buffer(self.local_loss_ema, sample_ids, local_losses, previously_seen)

        self.sample_seen[sample_ids] = True
        self.seen_counts[sample_ids] += 1

        self._enqueue(sample_ids)
        self._update_fn_stats(image_features, text_features, pids)
        self._update_topk_counts(sample_ids, image_features, text_features)
        return self.diagnostics()

    def _update_feature_buffer(self, buffer, sample_ids, values, seen):
        unseen = ~seen
        if unseen.any():
            buffer[sample_ids[unseen]] = values[unseen]
        if seen.any():
            beta = self.ema_beta
            ids = sample_ids[seen]
            buffer[ids] = F.normalize(
                beta * buffer[ids] + (1.0 - beta) * values[seen],
                dim=1,
                p=2,
            )

    def _update_loss_buffer(self, buffer, sample_ids, values, seen):
        values = values.detach().float().view(-1).to(buffer.device)
        unseen = ~seen
        if unseen.any():
            buffer[sample_ids[unseen]] = values[unseen]
        if seen.any():
            ids = sample_ids[seen]
            alpha = self.loss_ema_alpha
            buffer[ids] = alpha * buffer[ids] + (1.0 - alpha) * values[seen]

    def _enqueue(self, sample_ids):
        if self.queue_size <= 0:
            return
        ids = sample_ids.detach().long().to(self.queue_ids.device)
        for sample_id in ids:
            ptr = int(self.queue_ptr.item())
            self.queue_ids[ptr] = sample_id
            self.queue_ptr.fill_((ptr + 1) % self.queue_size)
            self.queue_count.fill_(min(int(self.queue_count.item()) + 1, self.queue_size))

    def _update_fn_stats(self, image_features, text_features, pids):
        if image_features.shape[0] < 2:
            return
        sim_mat = image_features @ text_features.t()
        pids_col = pids.view(-1, 1)
        pos_mask = torch.eq(pids_col, pids_col.t())
        neg_mask = ~pos_mask
        s_p = sim_mat[pos_mask]
        s_n = sim_mat[neg_mask]
        if s_p.numel() < 2 or s_n.numel() < 2:
            return

        batch_mu_pos = s_p.float().mean()
        batch_sigma_pos = s_p.float().std(unbiased=False).clamp_min(1e-4)
        batch_mu_neg = s_n.float().mean()
        batch_sigma_neg = s_n.float().std(unbiased=False).clamp_min(1e-4)
        if self.fn_stats_initialized.item() < 0.5:
            self.mu_pos.copy_(batch_mu_pos)
            self.sigma_pos.copy_(batch_sigma_pos)
            self.mu_neg.copy_(batch_mu_neg)
            self.sigma_neg.copy_(batch_sigma_neg)
            self.fn_stats_initialized.fill_(1.0)
            return

        beta = self.ema_beta
        self.mu_pos.mul_(beta).add_(batch_mu_pos, alpha=1.0 - beta)
        self.sigma_pos.mul_(beta).add_(batch_sigma_pos, alpha=1.0 - beta)
        self.mu_neg.mul_(beta).add_(batch_mu_neg, alpha=1.0 - beta)
        self.sigma_neg.mul_(beta).add_(batch_sigma_neg, alpha=1.0 - beta)

    def _update_topk_counts(self, sample_ids, image_features, text_features):
        if self.topk <= 0 or sample_ids.numel() < 2:
            return
        sim_mat = image_features @ text_features.t()
        k = min(self.topk, sim_mat.shape[1])
        topk_cols = torch.topk(sim_mat, k=k, dim=1).indices
        ids = sample_ids.detach().cpu().tolist()
        for row_idx, cols in enumerate(topk_cols.cpu().tolist()):
            image_id = int(ids[row_idx])
            for col_idx in cols:
                self.topk_pair_counts[(image_id, int(ids[col_idx]))] += 1

    @torch.no_grad()
    def refit_epoch(self, epoch: int) -> dict:
        """Refit global/local loss GMMs and update consensus labels."""
        if self.gmm_interval > 0 and epoch >= 0:
            if self.last_refit_epoch.item() >= 0 and (epoch - int(self.last_refit_epoch.item())) < self.gmm_interval:
                return self.diagnostics(prefix_epoch=epoch)
        self.last_refit_epoch.fill_(int(epoch))

        global_diag = self._refit_one(
            losses=self.global_loss_ema,
            seen_mask=self.sample_seen,
            clean_probs=self.global_clean_prob,
            fitted_flag=self.global_gmm_fitted,
            fallback_flag=self.global_gmm_fallback,
        )
        local_diag = self._refit_one(
            losses=self.local_loss_ema,
            seen_mask=self.local_seen,
            clean_probs=self.local_clean_prob,
            fitted_flag=self.local_gmm_fitted,
            fallback_flag=self.local_gmm_fallback,
        )
        self._update_consensus()

        diag = self.diagnostics(prefix_epoch=epoch)
        diag.update({f"global_{k}": v for k, v in global_diag.items()})
        diag.update({f"local_{k}": v for k, v in local_diag.items()})
        return diag

    def _refit_one(self, losses, seen_mask, clean_probs, fitted_flag, fallback_flag):
        seen_idx = seen_mask.nonzero(as_tuple=False).squeeze(-1)
        if seen_idx.numel() < self.min_gmm_samples:
            clean_probs.fill_(1.0)
            fitted_flag.fill_(0.0)
            fallback_flag.fill_(1.0)
            return self._empty_gmm_diag(float(seen_idx.numel()), fallback=1.0)

        data = losses[seen_idx].float()
        mu_0, sigma_0, pi_0, mu_1, sigma_1, pi_1 = self._fit_1d_gmm_em(data)
        if mu_0.item() <= mu_1.item():
            mu_clean, sigma_clean, pi_clean = mu_0, sigma_0, pi_0
            mu_noisy, sigma_noisy, pi_noisy = mu_1, sigma_1, pi_1
        else:
            mu_clean, sigma_clean, pi_clean = mu_1, sigma_1, pi_1
            mu_noisy, sigma_noisy, pi_noisy = mu_0, sigma_0, pi_0

        separation = (mu_noisy - mu_clean).abs() / (sigma_clean + sigma_noisy + 1e-8)
        fallback = separation.item() < self.min_gmm_separation
        if fallback:
            clean_probs.fill_(1.0)
            fallback_flag.fill_(1.0)
        else:
            log_p_clean = self._log_gaussian(data, mu_clean, sigma_clean) + torch.log(pi_clean + 1e-12)
            log_p_noisy = self._log_gaussian(data, mu_noisy, sigma_noisy) + torch.log(pi_noisy + 1e-12)
            clean_prob = (log_p_clean - torch.logaddexp(log_p_clean, log_p_noisy)).exp()
            clean_probs.fill_(1.0)
            clean_probs[seen_idx] = clean_prob.to(clean_probs.dtype)
            fallback_flag.fill_(0.0)
        fitted_flag.fill_(1.0)

        return {
            "separation": separation.item(),
            "mu_clean": mu_clean.item(),
            "mu_noisy": mu_noisy.item(),
            "sigma_clean": sigma_clean.item(),
            "sigma_noisy": sigma_noisy.item(),
            "pi_clean": pi_clean.item(),
            "pi_noisy": pi_noisy.item(),
            "fallback": 1.0 if fallback else 0.0,
            "n_samples_used": float(seen_idx.numel()),
        }

    @staticmethod
    def _empty_gmm_diag(n_samples, fallback):
        return {
            "separation": 0.0,
            "mu_clean": 0.0,
            "mu_noisy": 0.0,
            "sigma_clean": 0.0,
            "sigma_noisy": 0.0,
            "pi_clean": 1.0,
            "pi_noisy": 0.0,
            "fallback": fallback,
            "n_samples_used": n_samples,
        }

    def _update_consensus(self):
        self.consensus_label.fill_(-1)
        if self.global_gmm_fitted.item() < 0.5 or self.local_gmm_fitted.item() < 0.5:
            return
        if self.global_gmm_fallback.item() > 0.5 or self.local_gmm_fallback.item() > 0.5:
            return
        valid = self.sample_seen & self.local_seen & (self.seen_counts >= self.min_seen_epochs)
        global_clean = self.global_clean_prob >= self.clean_threshold
        local_clean = self.local_clean_prob >= self.clean_threshold
        confident_clean = valid & global_clean & local_clean
        confident_noisy = valid & (~global_clean) & (~local_clean)
        self.consensus_label[confident_clean] = 1
        self.consensus_label[confident_noisy] = 0

    def get_fn_stats_dict(self):
        if self.fn_stats_initialized.item() < 0.5:
            return None
        return {
            "mu_pos": self.mu_pos.item(),
            "sigma_pos": self.sigma_pos.item(),
            "mu_neg": self.mu_neg.item(),
            "sigma_neg": self.sigma_neg.item(),
            "fn_prior": self.fn_prior,
        }

    def get_consensus_for_batch(self, sample_ids: torch.Tensor) -> torch.Tensor:
        sample_ids = sample_ids.detach().long().view(-1).to(self.consensus_label.device)
        valid = (sample_ids >= 0) & (sample_ids < self.num_train_samples)
        labels = torch.full_like(sample_ids, -1)
        labels[valid] = self.consensus_label[sample_ids[valid]]
        return labels

    def get_seen_mask_for_batch(self, sample_ids: torch.Tensor) -> torch.Tensor:
        sample_ids = sample_ids.detach().long().view(-1).to(self.sample_seen.device)
        valid = (sample_ids >= 0) & (sample_ids < self.num_train_samples)
        seen = torch.zeros_like(sample_ids, dtype=torch.bool)
        seen[valid] = self.sample_seen[sample_ids[valid]]
        return seen

    def diagnostics(self, prefix_epoch: int | None = None) -> dict:
        seen_frac = self.sample_seen.float().mean().item()
        local_seen_frac = self.local_seen.float().mean().item()
        confident_clean = (self.consensus_label == 1).float().mean().item()
        confident_noisy = (self.consensus_label == 0).float().mean().item()
        uncertain = (self.consensus_label < 0).float().mean().item()
        diag = {
            "seen_frac": seen_frac,
            "local_seen_frac": local_seen_frac,
            "queue_count": float(self.queue_count.item()),
            "fn_stats_ready": float(self.fn_stats_initialized.item()),
            "fn_mu_pos": float(self.mu_pos.item()),
            "fn_mu_neg": float(self.mu_neg.item()),
            "consensus_clean_frac": confident_clean,
            "consensus_noisy_frac": confident_noisy,
            "consensus_uncertain_frac": uncertain,
            "topk_pair_count": float(len(self.topk_pair_counts)),
            "global_gmm_fallback": float(self.global_gmm_fallback.item()),
            "local_gmm_fallback": float(self.local_gmm_fallback.item()),
        }
        if prefix_epoch is not None:
            diag["epoch"] = float(prefix_epoch)
        return diag

    @staticmethod
    def _log_gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)

    @staticmethod
    def _fit_1d_gmm_em(data: torch.Tensor, n_iter: int = 50, tol: float = 1e-6):
        data = data.detach().float().view(-1)
        n = data.numel()
        q25, q75 = torch.quantile(data, torch.tensor([0.25, 0.75], device=data.device))
        mu_0 = q25.clone()
        mu_1 = q75.clone()
        sigma_0 = data.std(unbiased=False).clamp_min(1e-4)
        sigma_1 = sigma_0.clone()
        pi_0 = torch.tensor(0.5, device=data.device)
        pi_1 = torch.tensor(0.5, device=data.device)
        prev_log_lik = torch.tensor(-float("inf"), device=data.device)

        for _ in range(n_iter):
            log_p_0 = EvidenceMemoryBank._log_gaussian(data, mu_0, sigma_0) + torch.log(pi_0 + 1e-12)
            log_p_1 = EvidenceMemoryBank._log_gaussian(data, mu_1, sigma_1) + torch.log(pi_1 + 1e-12)
            log_denom = torch.logaddexp(log_p_0, log_p_1)
            r_0 = (log_p_0 - log_denom).exp()
            r_1 = 1.0 - r_0
            n_0 = r_0.sum().clamp_min(1e-6)
            n_1 = r_1.sum().clamp_min(1e-6)

            mu_0 = (r_0 * data).sum() / n_0
            mu_1 = (r_1 * data).sum() / n_1
            sigma_0 = ((r_0 * (data - mu_0) ** 2).sum() / n_0).sqrt().clamp_min(1e-4)
            sigma_1 = ((r_1 * (data - mu_1) ** 2).sum() / n_1).sqrt().clamp_min(1e-4)
            pi_0 = n_0 / n
            pi_1 = n_1 / n

            log_lik = log_denom.sum()
            if (log_lik - prev_log_lik).abs() < tol:
                break
            prev_log_lik = log_lik

        return mu_0, sigma_0, pi_0, mu_1, sigma_1, pi_1

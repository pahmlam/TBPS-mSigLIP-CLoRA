import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_constrative(
    image_features,
    text_features,
    sim_targets,
    logit_scale,
    logit_bias,
    use_sigmoid,
):
    """
    Compute contrastive loss for image-text pairs.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    logit_t2i = logit_scale * text_features @ image_features.t() + logit_bias
    logit_i2t = logit_scale * image_features @ text_features.t() + logit_bias

    if use_sigmoid:
        loglik = F.logsigmoid(logit_t2i * sim_targets)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
    else:
        loss_i2t = -torch.sum(F.log_softmax(logit_i2t, dim=1) * sim_targets, dim=1)
        loss_t2i = -torch.sum(F.log_softmax(logit_t2i, dim=1) * sim_targets, dim=1)
        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2

    return loss


def compute_simclr(
    image_features_1,
    image_features_2,
    temperature=0.07,
):
    """
    Contrastive learning loss using SimCLR.
    """
    device = image_features_1.device
    batch_size = image_features_1.shape[0]

    image_features_1 = F.normalize(image_features_1, dim=-1, p=2)
    image_features_2 = F.normalize(image_features_2, dim=-1, p=2)

    labels = torch.arange(start=0, end=batch_size, device=device)

    sim_ab = (image_features_1 @ image_features_2.t()) / temperature
    sim_ba = sim_ab.t()

    mask = torch.where(F.one_hot(labels, batch_size) == 0, 0, float("-inf"))
    sim_aa = (image_features_1 @ image_features_1.t()) / temperature + mask
    sim_bb = (image_features_2 @ image_features_2.t()) / temperature + mask

    sim_a = torch.cat((sim_ab, sim_aa), dim=1)
    sim_b = torch.cat((sim_ba, sim_bb), dim=1)

    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    return (loss_a + loss_b) / 2


def compute_citc(
    image_features,
    text_features,
    logit_scale,
    logit_bias,
    inmodal_weight,
    intermodal_weight,
):
    """
    Compute cyclic image-text contrastive loss.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_i2i = logit_scale * image_features @ image_features.t() + logit_bias
    sim_t2t = logit_scale * text_features @ text_features.t() + logit_bias

    sim_i2t = logit_scale * image_features @ text_features.t() + logit_bias
    sim_t2i = sim_i2t.t()

    inmodal_cyclic_loss = (sim_i2i - sim_t2t).square().mean() / (
        logit_scale * logit_scale
    )
    intermodal_cyclic_loss = (sim_i2t - sim_t2i).square().mean() / (
        logit_scale * logit_scale
    )

    return (
        inmodal_weight * inmodal_cyclic_loss
        + intermodal_weight * intermodal_cyclic_loss
    )


def compute_cross_modal_circle(image_features, text_features, pids, m=0.25, gamma=128):
    """
    Circle Loss between 2 modalities.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_mat = torch.matmul(image_features, text_features.t())

    pids = pids.view(-1, 1)

    pos_mask = torch.eq(pids, pids.t()).float()
    neg_mask = 1 - pos_mask

    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]

    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + m, min=0.)

    delta_p = 1 - m
    delta_n = m

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(
        torch.logsumexp(logit_p, dim=0) +
        torch.logsumexp(logit_n, dim=0)
    )

    return loss


def _bayesian_fn_prob(
    s_n: torch.Tensor,
    mu_pos: float,
    sigma_pos: float,
    mu_neg: float,
    sigma_neg: float,
    fn_prior: float,
) -> torch.Tensor:
    """
    Bayesian posterior P(FN | s) for each negative similarity.

    P(FN | s) = p * f+(s) / [p * f+(s) + (1-p) * f-(s)]

    where f+, f- are Gaussian likelihoods and p is the prior P(FN).
    Computed in log-space for numerical stability.

    Args:
        s_n: (Nn,) negative-pair similarities
        mu_pos, sigma_pos: Gaussian parameters of the positive distribution
        mu_neg, sigma_neg: Gaussian parameters of the negative distribution
        fn_prior: prior probability of false negatives among negatives

    Returns:
        (Nn,) tensor of P(FN | s_n_i) values in [0, 1].
    """
    sigma_pos = max(sigma_pos, 1e-6)
    sigma_neg = max(sigma_neg, 1e-6)

    # log N(s; mu, sigma^2) = -0.5 * ((s - mu)/sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    log_const = -0.5 * math.log(2 * math.pi)
    log_f_pos = -0.5 * ((s_n - mu_pos) / sigma_pos) ** 2 - math.log(sigma_pos) + log_const
    log_f_neg = -0.5 * ((s_n - mu_neg) / sigma_neg) ** 2 - math.log(sigma_neg) + log_const

    log_p = math.log(max(fn_prior, 1e-12))
    log_1mp = math.log(max(1.0 - fn_prior, 1e-12))

    # log(numerator) and log(denominator); log-sum-exp for denom.
    log_num = log_p + log_f_pos
    log_other = log_1mp + log_f_neg
    log_denom = torch.logaddexp(log_num, log_other)

    return (log_num - log_denom).exp().clamp(0.0, 1.0)


def compute_noise_aware_circle(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pids: torch.Tensor,
    m: float = 0.25,
    gamma: float = 128,
    fn_stats: dict | None = None,
    clean_weights: torch.Tensor | None = None,
    epsilon_n: float = 0.1,
    epsilon_p: float = 0.2,
) -> tuple:
    """
    Noise-Aware Circle Loss (Idea C — unified FN + FP handling).

    When both `fn_stats` and `clean_weights` are None, this function produces
    the SAME loss as `compute_cross_modal_circle()` — it degenerates to the
    vanilla Circle Loss. This is important for curriculum scheduling: the same
    function can be called for the entire NACIR ramp, with detection activating
    purely through the arguments.

    Args:
        image_features, text_features: (B, D) features (will be L2-normalized).
        pids: (B,) long tensor of person IDs.
        m, gamma: Circle Loss hyperparameters (margin, scale).
        fn_stats: None OR dict with keys {mu_pos, sigma_pos, mu_neg, sigma_neg, fn_prior}.
            When provided, alpha_n is softened by (1 - P_fn).
        clean_weights: None OR (B,) float tensor of per-sample clean probabilities.
            When provided, alpha_p is softened by min(w[i], w[j]) per positive pair.
        epsilon_n: floor for FN suppression: alpha_n *= max(1 - P_fn, epsilon_n).
        epsilon_p: floor for FP suppression: alpha_p *= max(pair_w, epsilon_p).

    Returns:
        loss: scalar tensor (requires_grad).
        diagnostics: dict with:
            's_p', 's_n': detached flattened similarities (for EMA updates)
            'per_sample_loss': (B,) detached per-sample Circle Loss (for GMM tracking)
            'fn_prob_mean': mean P(FN) over negatives (0.0 if FN disabled)
            'clean_weight_mean': mean clean weight over batch (1.0 if FP disabled)
            'alpha_n_scale_mean': mean FN-softening factor applied
            'alpha_p_scale_mean': mean FP-softening factor applied
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_mat = torch.matmul(image_features, text_features.t())  # (B, B)
    B = sim_mat.shape[0]

    pids_col = pids.view(-1, 1)
    pos_mask = torch.eq(pids_col, pids_col.t())  # (B, B) bool
    neg_mask = ~pos_mask

    # Index tensors for flattened positive/negative pairs.
    pos_indices = pos_mask.nonzero(as_tuple=False)  # (Np, 2)
    neg_indices = neg_mask.nonzero(as_tuple=False)  # (Nn, 2)

    s_p = sim_mat[pos_indices[:, 0], pos_indices[:, 1]]  # (Np,)
    s_n = sim_mat[neg_indices[:, 0], neg_indices[:, 1]]  # (Nn,)

    diagnostics = {
        "s_p": s_p.detach(),
        "s_n": s_n.detach(),
        "per_sample_loss": torch.zeros(B, device=sim_mat.device),
        "fn_prob_mean": 0.0,
        "clean_weight_mean": 1.0,
        "alpha_n_scale_mean": 1.0,
        "alpha_p_scale_mean": 1.0,
    }

    if s_p.numel() == 0 or s_n.numel() == 0:
        return (
            torch.tensor(0.0, device=image_features.device, requires_grad=True),
            diagnostics,
        )

    # ------------------ Vanilla alpha_p, alpha_n ------------------
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.0)  # (Np,)
    alpha_n = torch.clamp_min(s_n.detach() + m, min=0.0)  # (Nn,)

    delta_p = 1 - m
    delta_n = m

    # ------------------ FN softening (negative branch) ------------------
    fn_scale = torch.ones_like(alpha_n)
    if fn_stats is not None:
        fn_probs = _bayesian_fn_prob(
            s_n.detach(),
            mu_pos=fn_stats["mu_pos"],
            sigma_pos=fn_stats["sigma_pos"],
            mu_neg=fn_stats["mu_neg"],
            sigma_neg=fn_stats["sigma_neg"],
            fn_prior=fn_stats["fn_prior"],
        )
        fn_scale = torch.clamp_min(1.0 - fn_probs, min=epsilon_n)
        diagnostics["fn_prob_mean"] = fn_probs.mean().item()
        diagnostics["alpha_n_scale_mean"] = fn_scale.mean().item()

    alpha_n_tilde = alpha_n * fn_scale

    # ------------------ FP softening (positive branch) ------------------
    fp_scale = torch.ones_like(alpha_p)
    if clean_weights is not None:
        clean_weights = clean_weights.to(sim_mat.device)
        # For positive pair (i, j), use min(w[i], w[j]) — both samples must be
        # clean for the pair to get full weight. Conservative vs product.
        pair_w = torch.minimum(
            clean_weights[pos_indices[:, 0]],
            clean_weights[pos_indices[:, 1]],
        )
        fp_scale = torch.clamp_min(pair_w, min=epsilon_p)
        diagnostics["clean_weight_mean"] = clean_weights.mean().item()
        diagnostics["alpha_p_scale_mean"] = fp_scale.mean().item()

    alpha_p_tilde = alpha_p * fp_scale

    # ------------------ Aggregate loss (identical structure to vanilla) ------------------
    logit_p = -gamma * alpha_p_tilde * (s_p - delta_p)
    logit_n = gamma * alpha_n_tilde * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(
        torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)
    )

    # ------------------ Per-sample Circle Loss (detached, for GMM tracking) ------------------
    # Row-wise decomposition: for sample i, compute its row's Circle Loss contribution.
    # This is used ONLY for FP detection; it does not affect gradient.
    with torch.no_grad():
        sim_detached = sim_mat.detach()
        alpha_p_mat = torch.clamp_min(-sim_detached + 1 + m, min=0.0)
        alpha_n_mat = torch.clamp_min(sim_detached + m, min=0.0)
        logit_p_mat = -gamma * alpha_p_mat * (sim_detached - delta_p)
        logit_n_mat = gamma * alpha_n_mat * (sim_detached - delta_n)

        # Mask out non-positive / non-negative entries with -inf before logsumexp.
        neg_inf = torch.finfo(sim_detached.dtype).min
        masked_logit_p = torch.where(pos_mask, logit_p_mat, torch.full_like(logit_p_mat, neg_inf))
        masked_logit_n = torch.where(neg_mask, logit_n_mat, torch.full_like(logit_n_mat, neg_inf))

        # A row with no positives/negatives would have all -inf → logsumexp is -inf.
        # Replace those rows with 0 so they contribute 0 loss (no signal from that sample).
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)

        per_row_lp = torch.where(
            has_pos,
            torch.logsumexp(masked_logit_p, dim=1),
            torch.zeros(B, device=sim_detached.device),
        )
        per_row_ln = torch.where(
            has_neg,
            torch.logsumexp(masked_logit_n, dim=1),
            torch.zeros(B, device=sim_detached.device),
        )
        diagnostics["per_sample_loss"] = F.softplus(per_row_lp + per_row_ln)

    return loss, diagnostics

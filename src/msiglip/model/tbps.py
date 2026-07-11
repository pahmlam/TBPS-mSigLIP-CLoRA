import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from msiglip.model import objectives

class TBPS(nn.Module):
    def __init__(self, config, backbone, num_train_samples: int = 0):
        super().__init__()
        self.config = config

        self.backbone = backbone
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.embed_dim = config.backbone.embedding_dim

        self.use_sigmoid = config.backbone.use_sigmoid
        if not hasattr(self.backbone, "logit_bias"):
            self.backbone.logit_bias = 0

        self.nitc_mvs_mode = str(config.loss.get("nitc_mvs_mode", "baseline"))
        valid_nitc_mvs_modes = {"baseline", "direct"}
        if self.nitc_mvs_mode not in valid_nitc_mvs_modes:
            raise ValueError(
                "loss.nitc_mvs_mode must be one of "
                f"{sorted(valid_nitc_mvs_modes)}, got {self.nitc_mvs_mode!r}."
            )
        logger.info(f"N-ITC MVS mode: {self.nitc_mvs_mode}")

        self.contain_visual_projection, self.contain_text_projection = (
            self.check_contain_projection()
        )
        if config.loss.get("SS", None):
            self.simclr_mlp = self._build_mlp(
                self.embed_dim, self.embed_dim, self.embed_dim
            )

        mneb_enabled = bool(config.loss.get("MNEB", False))
        nacir_enabled = bool(config.loss.get("NACIR", False))
        if mneb_enabled and nacir_enabled:
            raise ValueError("MNEB and NACIR are mutually exclusive; enable only one.")

        # --- Noise-Aware Circle Loss state (Idea C) ---
        # Only instantiate when NACIR is enabled; otherwise stays None to avoid
        # checkpointing unused buffers.
        self.noise_state = None
        if nacir_enabled:
            if num_train_samples <= 0:
                raise ValueError(
                    "NACIR requires num_train_samples > 0 to size per-sample buffers. "
                    "Pass it from LitTBPS via trainer.py."
                )
            from msiglip.model.noise_aware import NoiseAwareCircleState

            nacir_cfg = config.loss.get("nacir_config", {})
            self.noise_state = NoiseAwareCircleState(num_train_samples, nacir_cfg)
            logger.info(
                f"NACIR enabled: num_train_samples={num_train_samples}, "
                f"fn_detector={self.noise_state.fn_detector}, "
                f"fn_enable_epoch={self.noise_state.fn_enable_epoch}, "
                f"fp_enable_epoch={self.noise_state.fp_enable_epoch}"
            )

        # --- MNEB-HN state ---
        # Evidence collection and auxiliary losses are optional and stay outside
        # the vanilla loss path unless loss.MNEB is explicitly enabled.
        self.evidence_bank = None
        if mneb_enabled:
            if num_train_samples <= 0:
                raise ValueError(
                    "MNEB requires num_train_samples > 0 to size evidence buffers. "
                    "Pass it from LitTBPS via trainer.py."
                )
            from msiglip.model.evidence_bank import EvidenceMemoryBank

            mneb_cfg = config.loss.get("mneb_config", {})
            self.evidence_bank = EvidenceMemoryBank(
                num_train_samples=num_train_samples,
                embed_dim=self.embed_dim,
                config=mneb_cfg,
            )
            logger.info(
                f"MNEB-HN enabled: num_train_samples={num_train_samples}, "
                f"queue_size={self.evidence_bank.queue_size}, "
                f"topk={self.evidence_bank.topk}, "
                f"fnm_enable_epoch={self.evidence_bank.fnm_enable_epoch}, "
                f"rde_enable_epoch={self.evidence_bank.rde_enable_epoch}"
            )

    def _build_mlp(self, in_dim=512, mlp_dim=128, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )

    def check_contain_projection(self):
        """
        Check if the backbone contains visual and text projection.
        """
        contain_visual_projection = hasattr(self.backbone, "visual_projection")
        logger.info(
            f"Contain visual projection: {contain_visual_projection}"
        ) if contain_visual_projection else None
        contain_text_projection = hasattr(self.backbone, "text_projection")
        logger.info(
            f"Contain text projection: {contain_text_projection}"
        ) if contain_text_projection else None
        return contain_visual_projection, contain_text_projection

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _mneb_cfg(self, name, default=None):
        mneb_cfg = self.config.loss.get("mneb_config", {})
        return self._cfg_get(mneb_cfg, name, {} if default is None else default)

    def _image_grid_hw(self):
        image_size = self.config.get("img_size", None)
        if image_size is None:
            return None
        patch_size = self.config.backbone.vision_config.patch_size
        return (
            int(image_size[0] // patch_size),
            int(image_size[1] // patch_size),
        )

    @staticmethod
    def _mean_pool_text_tokens(text_tokens, attention_mask):
        mask = attention_mask.to(text_tokens.device).bool()
        mask_float = mask.float().unsqueeze(-1)
        denom = mask_float.sum(dim=1).clamp_min(1.0)
        return (text_tokens * mask_float).sum(dim=1) / denom

    @staticmethod
    def _as_ret_tensor(value, device):
        return torch.tensor(float(value), device=device)

    def encode_image(self, image, return_last_hidden=False):
        """
        Encodes an image into a tensor.

        Args:
            image (PIL.Image): Image to be encoded.
            return_last_hidden (bool): Whether to return the last hidden state.

        Returns:
            torch.Tensor: The encoded image, use pooler_output as the image embedding."""
        x = self.vision_model(image)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "visual_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.visual_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.visual_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def encode_text(self, text, return_last_hidden=False):
        """
        Encodes text into a tensor.

        Args:
            text (dict): Text to be encoded, containing the keys "input_ids" and "attention_mask".
            return_last_hidden (bool): Whether to return the last hidden state.
        Returns:
            torch.Tensor: The encoded text, use pooler_output as the sentence embedding."""
        # text will be a dict that has forms {"input_ids": ..., "attention_mask": ...}
        x = self.text_model(**text)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "text_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.text_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.text_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def prepare_sim_targets(self, pids, use_sigmoid=True):
        """
        Prepare similarity targets for constrative learning.

        Args:
            pids (torch.Tensor): Tensor containing the person IDs.
        Returns:
            torch.Tensor: Tensor containing the similarity targets.
        """
        sim_targets = torch.eq(pids.view(-1, 1), pids.view(1, -1)).float()
        if use_sigmoid:
            sim_targets = (
                -torch.ones_like(sim_targets) + 2 * sim_targets
            )  # -1 if different, 1 if same
            return sim_targets

        return sim_targets / sim_targets.sum(
            dim=1, keepdim=True
        )

    def _build_mneb_fn_candidates(
        self,
        global_score_mat,
        local_score_mat,
        pids,
        fn_stats,
        fnm_cfg,
    ):
        device = global_score_mat.device
        empty_mask = torch.zeros_like(global_score_mat, dtype=torch.bool)
        empty_probs = torch.zeros_like(global_score_mat)
        diag = {
            "fn_aux_prob_mean": 0.0,
            "fn_aux_prob_max": 0.0,
            "fn_aux_candidate_frac": 0.0,
            "fn_aux_selected_frac": 0.0,
            "fn_aux_local_agreement": 0.0,
        }
        if fn_stats is None:
            return empty_mask, empty_probs, diag

        pids = pids.to(device)
        pids_col = pids.view(-1, 1)
        pos_mask = torch.eq(pids_col, pids_col.t())
        neg_mask = ~pos_mask
        neg_indices = neg_mask.nonzero(as_tuple=False)
        if neg_indices.numel() == 0:
            return empty_mask, empty_probs, diag

        s_n = global_score_mat.detach()[neg_indices[:, 0], neg_indices[:, 1]]
        fn_probs = objectives._bayesian_fn_prob(
            s_n=s_n,
            mu_pos=fn_stats["mu_pos"],
            sigma_pos=fn_stats["sigma_pos"],
            mu_neg=fn_stats["mu_neg"],
            sigma_neg=fn_stats["sigma_neg"],
            fn_prior=fn_stats["fn_prior"],
        )
        prob_mat = empty_probs.clone()
        prob_mat[neg_indices[:, 0], neg_indices[:, 1]] = fn_probs.to(prob_mat.dtype)
        diag["fn_aux_prob_mean"] = float(fn_probs.mean().item())
        diag["fn_aux_prob_max"] = float(fn_probs.max().item())

        theta_fn = float(self._cfg_get(fnm_cfg, "theta_fn", 0.8))
        min_pos_neg_gap = float(self._cfg_get(fnm_cfg, "min_pos_neg_gap", 0.0))
        max_candidate_frac = float(self._cfg_get(fnm_cfg, "max_candidate_frac", 0.02))
        local_agreement = bool(self._cfg_get(fnm_cfg, "local_agreement", True))

        mu_pos = float(fn_stats["mu_pos"])
        sigma_pos = max(float(fn_stats["sigma_pos"]), 1e-6)
        mu_neg = float(fn_stats["mu_neg"])
        candidate = (
            (fn_probs >= theta_fn)
            & (s_n >= (mu_pos - sigma_pos))
            & ((mu_pos - mu_neg) > min_pos_neg_gap)
        )

        if local_agreement and local_score_mat is not None:
            local_detached = local_score_mat.detach()
            local_pos = local_detached[pos_mask]
            if local_pos.numel() > 1:
                local_threshold = local_pos.mean() - local_pos.std(unbiased=False).clamp_min(1e-6)
                local_pairs = local_detached[neg_indices[:, 0], neg_indices[:, 1]]
                candidate = candidate & (local_pairs >= local_threshold)
                diag["fn_aux_local_agreement"] = 1.0
            else:
                candidate = torch.zeros_like(candidate)

        diag["fn_aux_candidate_frac"] = float(candidate.float().mean().item())
        selected = torch.zeros_like(candidate)
        if candidate.any() and max_candidate_frac > 0:
            max_selected = max(1, int(max_candidate_frac * candidate.numel()))
            candidate_indices = candidate.nonzero(as_tuple=False).squeeze(1)
            topk = min(max_selected, candidate_indices.numel())
            topk_indices = torch.topk(fn_probs[candidate_indices], k=topk).indices
            selected[candidate_indices[topk_indices]] = True

        candidate_mask = empty_mask.clone()
        candidate_mask[neg_indices[:, 0], neg_indices[:, 1]] = selected
        diag["fn_aux_selected_frac"] = float(selected.float().mean().item())
        return candidate_mask, prob_mat, diag

    def forward(self, batch, current_epoch=None):
        ret = dict()

        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        logit_scale = self.backbone.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        part_align_enabled = self.config.loss.get("PART_ALIGN", None)
        mneb_enabled = self.evidence_bank is not None
        mneb_evidence_cfg = self._mneb_cfg("evidence_bank")
        mneb_evidence_enabled = bool(self._cfg_get(mneb_evidence_cfg, "enabled", True))
        mneb_fnm_cfg = self._mneb_cfg("fnm_aux")
        mneb_rde_cfg = self._mneb_cfg("rde_aux")
        mneb_fnm_needs_local = bool(self._cfg_get(mneb_fnm_cfg, "enabled", False)) and bool(
            self._cfg_get(mneb_fnm_cfg, "local_agreement", True)
        )
        mneb_needs_local = mneb_enabled and (
            mneb_evidence_enabled
            or bool(self._cfg_get(mneb_rde_cfg, "enabled", False))
            or mneb_fnm_needs_local
        )
        return_last_hidden = bool(part_align_enabled or mneb_needs_local)
        if return_last_hidden:
            image_pooler_output, image_last_hidden = self.encode_image(
                images, return_last_hidden=True
            )
            caption_pooler_output, text_last_hidden = self.encode_text(
                caption_input, return_last_hidden=True
            )
        else:
            image_pooler_output = self.encode_image(images)
            caption_pooler_output = self.encode_text(caption_input)
            image_last_hidden = None
            text_last_hidden = None

        ret.update({"temperature": self.backbone.logit_scale})
        ret.update({"bias": self.backbone.logit_bias})

        # --- A. SimCLR (Self-Supervised) ---
        if self.config.loss.get("SS", None):
            ss_images1_embed = self.simclr_mlp(self.encode_image(batch["ss_images1"]))
            ss_images2_embed = self.simclr_mlp(self.encode_image(batch["ss_images2"]))
            ss_loss = (
                objectives.compute_simclr(
                    ss_images1_embed,
                    ss_images2_embed,
                    self.config.loss.simclr_temperature,
                )
                * self.config.loss.ss_loss_weight
            )
            ret.update({"ss_loss": ss_loss})

        # --- CURRICULUM WEIGHT CALCULATION ---
        if current_epoch is None:
            current_epoch = 0

        T_warmup = 5
        T_ramp_len = 15
        T_stable = 20
        Lambda_target = self.config.loss.get("circle_loss_weight", 0.1)

        if current_epoch <= T_warmup:
            current_circle_weight = 0.0
        elif current_epoch <= T_stable:
            progress = (current_epoch - T_warmup) / T_ramp_len
            current_circle_weight = progress * Lambda_target
        else:
            current_circle_weight = Lambda_target

        ret.update({"circle_loss_weight": torch.tensor(current_circle_weight, device=image_pooler_output.device)})

        # --- B. N-ITC ---
        if self.config.loss.get("NITC", None):
            sim_targets = self.prepare_sim_targets(batch["pids"], self.use_sigmoid)

            nitc_loss = objectives.compute_constrative(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                sim_targets=sim_targets,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                use_sigmoid=self.use_sigmoid,
            )

            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                nitc_mvs_image_features = (
                    aug_images_features
                    if self.nitc_mvs_mode == "direct"
                    else image_pooler_output
                )
                augmented_nitc_loss = objectives.compute_constrative(
                    image_features=nitc_mvs_image_features,
                    text_features=caption_pooler_output,
                    sim_targets=sim_targets,
                    logit_scale=logit_scale,
                    use_sigmoid=self.use_sigmoid,
                    logit_bias=self.backbone.logit_bias,
                )
                nitc_loss = (nitc_loss + augmented_nitc_loss) / 2

            ret.update({"nitc_loss": nitc_loss * self.config.loss.nitc_loss_weight})

        # --- C. Cross-Modal Circle Loss (Curriculum) ---
        # Skip the vanilla CIR path when NACIR is enabled — NACIR replaces it.
        nacir_enabled = self.config.loss.get("NACIR", None) and self.noise_state is not None
        if self.config.loss.get("CIR", None) and not nacir_enabled and current_circle_weight > 0:
            circle_m = self.config.loss.get("circle_margin", 0.25)
            circle_gamma = self.config.loss.get("circle_gamma", 128)

            cm_circle_loss = objectives.compute_cross_modal_circle(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                pids=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )

            final_circle_loss = cm_circle_loss

            if self.config.loss.get("MVS", None):
                if 'aug_images_features' not in locals():
                    aug_images = batch["aug_images"]
                    aug_images_features = self.encode_image(aug_images)

                aug_cm_circle_loss = objectives.compute_cross_modal_circle(
                    image_features=aug_images_features,
                    text_features=caption_pooler_output,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma
                )
                final_circle_loss = (cm_circle_loss + aug_cm_circle_loss) / 2

            ret.update({"circle_loss": final_circle_loss * current_circle_weight})

        elif self.config.loss.get("CIR", None) and not nacir_enabled:
            ret.update({"circle_loss": torch.tensor(0.0, device=image_pooler_output.device, requires_grad=True)})

        # --- C2. Noise-Aware Circle Loss (Idea C) ---
        # Reuses the same curriculum weight as CIR. Detectors activate at
        # configured epoch thresholds (fn_enable_epoch, fp_enable_epoch).
        if nacir_enabled and current_circle_weight > 0:
            circle_m = self.config.loss.get("circle_margin", 0.25)
            circle_gamma = self.config.loss.get("circle_gamma", 128)

            # --- Gate detectors by curriculum epoch ---
            fn_active = (
                self.noise_state.fn_detector != "off"
                and current_epoch >= self.noise_state.fn_enable_epoch
            )
            fp_active = current_epoch >= self.noise_state.fp_enable_epoch

            fn_stats = self.noise_state.get_fn_stats_dict() if fn_active else None
            clean_weights = None
            if fp_active and "id" in batch:
                clean_weights = self.noise_state.get_clean_weights_for_batch(batch["id"])

            # --- Primary forward on clean images ---
            nacir_loss, diag = objectives.compute_noise_aware_circle(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                pids=batch["pids"],
                m=circle_m,
                gamma=circle_gamma,
                fn_stats=fn_stats,
                clean_weights=clean_weights,
                epsilon_n=self.noise_state.epsilon_n,
                epsilon_p=self.noise_state.epsilon_p,
                fn_safe_gate=self.noise_state.fn_safe_gate,
                fn_prob_threshold=self.noise_state.fn_prob_threshold,
                fn_pos_sigma_k=self.noise_state.fn_pos_sigma_k,
                fn_max_suppress_frac=self.noise_state.fn_max_suppress_frac,
                fn_min_pos_neg_gap=self.noise_state.fn_min_pos_neg_gap,
                fn_detector=self.noise_state.fn_detector,
                fn_mutual_topk=self.noise_state.fn_mutual_topk,
                fn_mutual_min_sim=self.noise_state.fn_mutual_min_sim,
            )

            final_nacir_loss = nacir_loss

            # --- MVS augmentation path (mirror of CIR MVS pattern) ---
            if self.config.loss.get("MVS", None):
                if 'aug_images_features' not in locals():
                    aug_images = batch["aug_images"]
                    aug_images_features = self.encode_image(aug_images)

                aug_nacir_loss, _ = objectives.compute_noise_aware_circle(
                    image_features=aug_images_features,
                    text_features=caption_pooler_output,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma,
                    fn_stats=fn_stats,
                    clean_weights=clean_weights,
                    epsilon_n=self.noise_state.epsilon_n,
                    epsilon_p=self.noise_state.epsilon_p,
                    fn_safe_gate=self.noise_state.fn_safe_gate,
                    fn_prob_threshold=self.noise_state.fn_prob_threshold,
                    fn_pos_sigma_k=self.noise_state.fn_pos_sigma_k,
                    fn_max_suppress_frac=self.noise_state.fn_max_suppress_frac,
                    fn_min_pos_neg_gap=self.noise_state.fn_min_pos_neg_gap,
                    fn_detector=self.noise_state.fn_detector,
                    fn_mutual_topk=self.noise_state.fn_mutual_topk,
                    fn_mutual_min_sim=self.noise_state.fn_mutual_min_sim,
                )
                final_nacir_loss = (nacir_loss + aug_nacir_loss) / 2

            ret.update({"nacir_loss": final_nacir_loss * current_circle_weight})

            # --- Update state (no_grad operations) ---
            if self.noise_state.fn_detector != "off":
                self.noise_state.update_ema_stats(diag["s_p"], diag["s_n"])
            if "id" in batch:
                self.noise_state.update_sample_losses(
                    batch["id"], diag["per_sample_loss"]
                )

            # --- Diagnostics for W&B logging ---
            # Prefix scalar diagnostics so they don't accidentally get summed
            # into total_loss (Lightning sums only keys ending in "loss").
            ret.update({
                "nacir_fn_prob_mean": torch.tensor(diag["fn_prob_mean"], device=image_pooler_output.device),
                "nacir_clean_weight_mean": torch.tensor(diag["clean_weight_mean"], device=image_pooler_output.device),
                "nacir_alpha_n_scale_mean": torch.tensor(diag["alpha_n_scale_mean"], device=image_pooler_output.device),
                "nacir_alpha_p_scale_mean": torch.tensor(diag["alpha_p_scale_mean"], device=image_pooler_output.device),
                "nacir_fn_selected_frac": torch.tensor(diag["fn_selected_frac"], device=image_pooler_output.device),
                "nacir_fn_gate_active": torch.tensor(diag["fn_gate_active"], device=image_pooler_output.device),
                "nacir_fn_prob_max": torch.tensor(diag["fn_prob_max"], device=image_pooler_output.device),
                "nacir_fn_prob_selected_mean": torch.tensor(diag["fn_prob_selected_mean"], device=image_pooler_output.device),
                "nacir_fn_candidate_frac": torch.tensor(diag["fn_candidate_frac"], device=image_pooler_output.device),
                "nacir_fn_selected_sim_mean": torch.tensor(diag["fn_selected_sim_mean"], device=image_pooler_output.device),
                "nacir_fn_detector_mutual": torch.tensor(diag["fn_detector_mutual"], device=image_pooler_output.device),
                "nacir_fn_active": torch.tensor(1.0 if fn_active else 0.0, device=image_pooler_output.device),
                "nacir_fp_active": torch.tensor(1.0 if fp_active else 0.0, device=image_pooler_output.device),
            })

        elif nacir_enabled:
            # Curriculum weight is 0 — still populate nacir_loss so the key exists
            # in the returned dict for consistent logging.
            ret.update({"nacir_loss": torch.tensor(0.0, device=image_pooler_output.device, requires_grad=True)})

        # --- C3. MNEB-HN evidence and optional auxiliary losses ---
        # MNEB never replaces or mutates Circle Loss. With aux losses disabled,
        # this block only collects/logs evidence and has no training effect.
        if mneb_enabled:
            device = image_pooler_output.device
            mneb_cfg = self.config.loss.get("mneb_config", {})
            fnm_cfg = self._cfg_get(mneb_cfg, "fnm_aux", {})
            rde_cfg = self._cfg_get(mneb_cfg, "rde_aux", {})
            fnm_enabled = bool(self._cfg_get(fnm_cfg, "enabled", False))
            rde_enabled = bool(self._cfg_get(rde_cfg, "enabled", False))
            fnm_active = fnm_enabled and current_epoch >= int(self._cfg_get(fnm_cfg, "enable_epoch", 15))
            rde_active = rde_enabled and current_epoch >= int(self._cfg_get(rde_cfg, "enable_epoch", 15))

            global_image_features = F.normalize(image_pooler_output, dim=1, p=2)
            global_text_features = F.normalize(caption_pooler_output, dim=1, p=2)
            global_score_mat = global_image_features @ global_text_features.t()
            global_losses = objectives.compute_branch_per_sample_contrastive_loss(
                score_mat=global_score_mat,
                pids=batch["pids"],
                temperature=self._cfg_get(rde_cfg, "temperature", 0.07),
            )

            local_score_mat = None
            local_losses = None
            local_image_features = None
            local_text_features = None
            if image_last_hidden is not None and text_last_hidden is not None:
                image_grid_hw = self._image_grid_hw()
                local_score_mat, local_active = objectives.compute_part_token_score_matrix(
                    image_tokens=image_last_hidden,
                    text_tokens=text_last_hidden,
                    attention_mask=batch["caption_attention_mask"],
                    num_parts=self.config.loss.get("part_align_num_parts", 4),
                    image_grid_hw=image_grid_hw,
                )
                local_losses = torch.zeros(
                    global_losses.shape[0],
                    device=device,
                    dtype=global_losses.dtype,
                )
                if local_active.any():
                    local_losses[local_active] = objectives.compute_branch_per_sample_contrastive_loss(
                        score_mat=local_score_mat[local_active][:, local_active],
                        pids=batch["pids"][local_active],
                        temperature=self._cfg_get(rde_cfg, "temperature", 0.07),
                    )
                local_image_features = image_last_hidden.mean(dim=1)
                local_text_features = self._mean_pool_text_tokens(
                    text_last_hidden,
                    batch["caption_attention_mask"],
                )

            ret.update({
                "mneb_enabled": self._as_ret_tensor(1.0, device),
                "mneb_fnm_active": self._as_ret_tensor(1.0 if fnm_active else 0.0, device),
                "mneb_rde_active": self._as_ret_tensor(1.0 if rde_active else 0.0, device),
            })

            if mneb_evidence_enabled and "id" in batch:
                mneb_diag = self.evidence_bank.update_batch(
                    sample_ids=batch["id"],
                    image_features=global_image_features,
                    text_features=global_text_features,
                    pids=batch["pids"],
                    global_losses=global_losses,
                    local_losses=local_losses,
                    local_image_features=local_image_features,
                    local_text_features=local_text_features,
                )
                ret.update({
                    f"mneb_{key}": self._as_ret_tensor(value, device)
                    for key, value in mneb_diag.items()
                    if isinstance(value, (int, float, bool))
                })
                ret.update({"mneb_has_ids": self._as_ret_tensor(1.0, device)})
            else:
                ret.update({"mneb_has_ids": self._as_ret_tensor(0.0, device)})

            if fnm_enabled:
                if fnm_active:
                    fn_stats = self.evidence_bank.get_fn_stats_dict()
                    fn_mask, fn_prob_mat, fn_diag = self._build_mneb_fn_candidates(
                        global_score_mat=global_score_mat,
                        local_score_mat=local_score_mat,
                        pids=batch["pids"],
                        fn_stats=fn_stats,
                        fnm_cfg=fnm_cfg,
                    )
                    fnm_aux_loss = objectives.compute_fnm_auxiliary_loss(
                        image_features=image_pooler_output,
                        text_features=caption_pooler_output,
                        pids=batch["pids"],
                        fn_candidate_mask=fn_mask,
                        fn_prob_matrix=fn_prob_mat,
                        theta_fn=self._cfg_get(fnm_cfg, "theta_fn", 0.8),
                        margin=self._cfg_get(fnm_cfg, "margin", 0.25),
                        temperature=self._cfg_get(fnm_cfg, "temperature", 0.07),
                    )
                    ret.update({
                        f"mneb_{key}": self._as_ret_tensor(value, device)
                        for key, value in fn_diag.items()
                    })
                else:
                    fnm_aux_loss = objectives._zero_loss_like(
                        image_pooler_output,
                        caption_pooler_output,
                    )

                fnm_weight = float(self._cfg_get(fnm_cfg, "weight", 0.05))
                ret.update({"fnm_aux_loss": fnm_aux_loss * fnm_weight})

            if rde_enabled:
                if rde_active and "id" in batch:
                    consensus_labels = self.evidence_bank.get_consensus_for_batch(batch["id"]).to(device)
                    rde_aux_loss = objectives.compute_rde_auxiliary_loss(
                        global_score_mat=global_score_mat,
                        local_score_mat=local_score_mat,
                        pids=batch["pids"],
                        consensus_labels=consensus_labels,
                        margin=self._cfg_get(rde_cfg, "margin", 0.2),
                        temperature=self._cfg_get(rde_cfg, "temperature", 0.07),
                    )
                    ret.update({
                        "mneb_batch_clean_frac": self._as_ret_tensor(
                            (consensus_labels == 1).float().mean().item(),
                            device,
                        ),
                        "mneb_batch_noisy_frac": self._as_ret_tensor(
                            (consensus_labels == 0).float().mean().item(),
                            device,
                        ),
                        "mneb_batch_uncertain_frac": self._as_ret_tensor(
                            (consensus_labels < 0).float().mean().item(),
                            device,
                        ),
                    })
                else:
                    rde_aux_loss = objectives._zero_loss_like(global_score_mat, local_score_mat)

                rde_weight = float(self._cfg_get(rde_cfg, "weight", 0.05))
                ret.update({"rde_aux_loss": rde_aux_loss * rde_weight})

        # --- D. C-ITC ---
        if self.config.loss.get("CITC", None):
            loss = objectives.compute_citc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                inmodal_weight=self.config.loss.citc_inmodal_weight,
                intermodal_weight=self.config.loss.citc_intermodal_weight,
            )
            ret.update({"citc_loss": loss * self.config.loss.citc_loss_weight})

        # --- E. Part-Token Local Alignment (training-only auxiliary loss) ---
        if part_align_enabled:
            part_target_weight = self.config.loss.get("part_align_loss_weight", 0.05)
            part_warmup_epoch = int(self.config.loss.get("part_align_warmup_epoch", 5))
            part_ramp_epoch = int(self.config.loss.get("part_align_ramp_epoch", 15))

            if current_epoch <= part_warmup_epoch:
                current_part_weight = 0.0
            elif part_ramp_epoch <= 0:
                current_part_weight = part_target_weight
            elif current_epoch <= part_warmup_epoch + part_ramp_epoch:
                progress = (current_epoch - part_warmup_epoch) / part_ramp_epoch
                current_part_weight = progress * part_target_weight
            else:
                current_part_weight = part_target_weight

            ret.update({
                "part_align_loss_weight": torch.tensor(
                    current_part_weight, device=image_pooler_output.device
                )
            })

            if current_part_weight > 0:
                image_size = self.config.get("img_size", None)
                patch_size = self.config.backbone.vision_config.patch_size
                image_grid_hw = None
                if image_size is not None:
                    image_grid_hw = (
                        int(image_size[0] // patch_size),
                        int(image_size[1] // patch_size),
                    )

                part_align_loss = objectives.compute_part_token_alignment(
                    image_tokens=image_last_hidden,
                    text_tokens=text_last_hidden,
                    attention_mask=batch["caption_attention_mask"],
                    pids=batch["pids"],
                    num_parts=self.config.loss.get("part_align_num_parts", 4),
                    temperature=self.config.loss.get("part_align_temperature", 0.07),
                    image_grid_hw=image_grid_hw,
                )
                ret.update({"part_align_loss": part_align_loss * current_part_weight})
            else:
                ret.update({
                    "part_align_loss": torch.tensor(
                        0.0,
                        device=image_pooler_output.device,
                        requires_grad=True,
                    )
                })

        return ret

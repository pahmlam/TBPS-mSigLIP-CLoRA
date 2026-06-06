import torch
import torch.nn as nn
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

        self.contain_visual_projection, self.contain_text_projection = (
            self.check_contain_projection()
        )
        if config.loss.get("SS", None):
            self.simclr_mlp = self._build_mlp(
                self.embed_dim, self.embed_dim, self.embed_dim
            )

        # --- Noise-Aware Circle Loss state (Idea C) ---
        # Only instantiate when NACIR is enabled; otherwise stays None to avoid
        # checkpointing unused buffers.
        self.noise_state = None
        if config.loss.get("NACIR", None):
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
                f"fn_enable_epoch={self.noise_state.fn_enable_epoch}, "
                f"fp_enable_epoch={self.noise_state.fp_enable_epoch}"
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

    def forward(self, batch, current_epoch=None):
        ret = dict()

        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        logit_scale = self.backbone.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        image_pooler_output = self.encode_image(images)
        caption_pooler_output = self.encode_text(caption_input)

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
                augmented_nitc_loss = objectives.compute_constrative(
                    image_features=image_pooler_output,
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
            fn_active = current_epoch >= self.noise_state.fn_enable_epoch
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
                )
                final_nacir_loss = (nacir_loss + aug_nacir_loss) / 2

            ret.update({"nacir_loss": final_nacir_loss * current_circle_weight})

            # --- Update state (no_grad operations) ---
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
                "nacir_fn_active": torch.tensor(1.0 if fn_active else 0.0, device=image_pooler_output.device),
                "nacir_fp_active": torch.tensor(1.0 if fp_active else 0.0, device=image_pooler_output.device),
            })

        elif nacir_enabled:
            # Curriculum weight is 0 — still populate nacir_loss so the key exists
            # in the returned dict for consistent logging.
            ret.update({"nacir_loss": torch.tensor(0.0, device=image_pooler_output.device, requires_grad=True)})

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

        return ret

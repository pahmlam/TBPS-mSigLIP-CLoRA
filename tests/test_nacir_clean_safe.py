import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.objectives import (  # noqa: E402
    _mutual_topk_negative_mask,
    compute_cross_modal_circle,
    compute_noise_aware_circle,
)
from msiglip.model.noise_aware import NoiseAwareCircleState  # noqa: E402


class NACIRCleanSafeTest(unittest.TestCase):
    def test_noop_matches_circle(self):
        torch.manual_seed(2400)
        image_features = torch.randn(8, 16)
        text_features = torch.randn(8, 16)
        pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        circle = compute_cross_modal_circle(image_features, text_features, pids)
        nacir, diag = compute_noise_aware_circle(image_features, text_features, pids)

        self.assertAlmostEqual(circle.item(), nacir.item(), places=5)
        self.assertEqual(diag["alpha_n_scale_mean"], 1.0)
        self.assertEqual(diag["alpha_p_scale_mean"], 1.0)

    def test_fn_gate_blocks_low_confidence_suppression(self):
        torch.manual_seed(2401)
        image_features = torch.randn(12, 16)
        text_features = torch.randn(12, 16)
        pids = torch.arange(12)
        fn_stats = {
            "mu_pos": 0.9,
            "sigma_pos": 0.05,
            "mu_neg": -0.2,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        _, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="bayesian",
            epsilon_n=0.85,
            fn_safe_gate=True,
            fn_prob_threshold=1.1,
        )

        self.assertEqual(diag["alpha_n_scale_mean"], 1.0)
        self.assertEqual(diag["fn_selected_frac"], 0.0)
        self.assertEqual(diag["fn_gate_active"], 0.0)

    def test_fn_gate_caps_suppressed_negative_fraction(self):
        image_features = torch.ones(20, 16)
        text_features = torch.ones(20, 16)
        pids = torch.arange(20)
        fn_stats = {
            "mu_pos": 1.0,
            "sigma_pos": 0.2,
            "mu_neg": -1.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        _, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="bayesian",
            epsilon_n=0.85,
            fn_safe_gate=True,
            fn_prob_threshold=0.8,
            fn_max_suppress_frac=0.05,
        )

        self.assertGreater(diag["fn_selected_frac"], 0.0)
        self.assertLessEqual(diag["fn_selected_frac"], 0.05 + 1e-6)
        self.assertLess(diag["alpha_n_scale_mean"], 1.0)

    def test_epsilon_n_is_floor_for_selected_negatives(self):
        image_features = torch.ones(20, 16)
        text_features = torch.ones(20, 16)
        pids = torch.arange(20)
        fn_stats = {
            "mu_pos": 1.0,
            "sigma_pos": 0.2,
            "mu_neg": -1.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        _, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="bayesian",
            epsilon_n=0.85,
            fn_safe_gate=True,
            fn_prob_threshold=0.8,
            fn_max_suppress_frac=1.0,
        )

        self.assertAlmostEqual(diag["alpha_n_scale_mean"], 0.85, places=5)
        self.assertAlmostEqual(diag["fn_selected_frac"], 1.0, places=5)

    def test_fn_gate_supports_fp16_features(self):
        image_features = torch.ones(20, 16, dtype=torch.float16)
        text_features = torch.ones(20, 16, dtype=torch.float16)
        pids = torch.arange(20)
        fn_stats = {
            "mu_pos": 1.0,
            "sigma_pos": 0.2,
            "mu_neg": -1.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        loss, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="bayesian",
            epsilon_n=0.85,
            fn_safe_gate=True,
            fn_prob_threshold=0.8,
            fn_max_suppress_frac=0.05,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diag["fn_selected_frac"], 0.0)

    def test_default_fn_detector_off_matches_circle_even_with_stats(self):
        torch.manual_seed(2403)
        image_features = torch.randn(8, 16)
        text_features = torch.randn(8, 16)
        pids = torch.arange(8)
        fn_stats = {
            "mu_pos": 0.8,
            "sigma_pos": 0.2,
            "mu_neg": -0.5,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        circle = compute_cross_modal_circle(image_features, text_features, pids)
        nacir, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
        )

        self.assertAlmostEqual(circle.item(), nacir.item(), places=5)
        self.assertEqual(diag["alpha_n_scale_mean"], 1.0)
        self.assertEqual(diag["fn_gate_active"], 0.0)

    def test_mutual_topk_noop_without_fn_stats(self):
        torch.manual_seed(2404)
        image_features = torch.randn(8, 16)
        text_features = torch.randn(8, 16)
        pids = torch.arange(8)

        circle = compute_cross_modal_circle(image_features, text_features, pids)
        nacir, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_detector="mutual_topk",
        )

        self.assertAlmostEqual(circle.item(), nacir.item(), places=5)
        self.assertEqual(diag["fn_detector_mutual"], 0.0)
        self.assertEqual(diag["fn_selected_frac"], 0.0)

    def test_mutual_topk_does_not_select_one_way_pair(self):
        sim_mat = torch.tensor(
            [
                [0.0, 0.90, 0.20, 0.10],
                [0.10, 0.0, 0.80, 0.20],
                [0.20, 0.95, 0.0, 0.85],
                [0.88, 0.10, 0.20, 0.0],
            ]
        )
        pids = torch.arange(4)
        pos_mask = torch.eq(pids.view(-1, 1), pids.view(1, -1))
        neg_indices = (~pos_mask).nonzero(as_tuple=False)
        fn_stats = {
            "mu_pos": 0.9,
            "sigma_pos": 0.1,
            "mu_neg": 0.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        candidate_mask = _mutual_topk_negative_mask(
            sim_mat=sim_mat,
            pos_mask=pos_mask,
            neg_indices=neg_indices,
            fn_stats=fn_stats,
            fn_pos_sigma_k=1.0,
            fn_min_pos_neg_gap=0.0,
            fn_mutual_topk=1,
            fn_mutual_min_sim=-1.0,
        )
        flat_index = ((neg_indices[:, 0] == 0) & (neg_indices[:, 1] == 1)).nonzero(
            as_tuple=False
        ).item()

        self.assertFalse(bool(candidate_mask[flat_index]))

    def test_mutual_topk_selects_reciprocal_high_sim_negative(self):
        image_features = torch.eye(4)
        text_features = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pids = torch.arange(4)
        fn_stats = {
            "mu_pos": 0.9,
            "sigma_pos": 0.1,
            "mu_neg": 0.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        _, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="mutual_topk",
            epsilon_n=0.85,
            fn_mutual_topk=1,
            fn_max_suppress_frac=1.0,
        )

        self.assertEqual(diag["fn_detector_mutual"], 1.0)
        self.assertGreater(diag["fn_candidate_frac"], 0.0)
        self.assertGreater(diag["fn_selected_frac"], 0.0)
        self.assertLess(diag["alpha_n_scale_mean"], 1.0)
        self.assertAlmostEqual(diag["fn_selected_sim_mean"], 1.0, places=5)

    def test_mutual_topk_caps_suppressed_negative_fraction(self):
        image_features = torch.eye(20)
        text_features = torch.roll(torch.eye(20), shifts=1, dims=0)
        pids = torch.arange(20)
        fn_stats = {
            "mu_pos": 0.9,
            "sigma_pos": 0.1,
            "mu_neg": 0.0,
            "sigma_neg": 0.2,
            "fn_prior": 0.5,
        }

        _, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            fn_stats=fn_stats,
            fn_detector="mutual_topk",
            fn_mutual_topk=1,
            fn_max_suppress_frac=0.02,
        )

        self.assertGreater(diag["fn_selected_frac"], 0.0)
        self.assertLessEqual(diag["fn_selected_frac"], 0.02 + 1e-6)

    def test_fp_clean_weights_path_is_unchanged(self):
        torch.manual_seed(2402)
        image_features = torch.randn(6, 16)
        text_features = torch.randn(6, 16)
        pids = torch.arange(6)
        clean_weights = torch.full((6,), 0.5)

        loss, diag = compute_noise_aware_circle(
            image_features,
            text_features,
            pids,
            clean_weights=clean_weights,
            epsilon_p=0.2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(diag["alpha_p_scale_mean"], 0.5, places=5)

    def test_noise_state_reads_fn_off_defaults(self):
        state = NoiseAwareCircleState(num_train_samples=4, config={})

        self.assertEqual(state.fn_detector, "off")
        self.assertEqual(state.fn_mutual_topk, 2)
        self.assertEqual(state.fn_mutual_min_sim, -1.0)
        self.assertEqual(state.fn_enable_epoch, 999)

    def test_noise_state_accepts_legacy_config_without_new_keys(self):
        state = NoiseAwareCircleState(
            num_train_samples=4,
            config={"fn_prior": 0.01, "epsilon_n": 0.85},
        )

        self.assertEqual(state.fn_detector, "off")
        self.assertEqual(state.fn_mutual_topk, 2)
        self.assertEqual(state.fn_mutual_min_sim, -1.0)
        self.assertEqual(state.fn_enable_epoch, 999)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.objectives import (  # noqa: E402
    compute_cross_modal_circle,
    compute_noise_aware_circle,
)


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
            epsilon_n=0.85,
            fn_safe_gate=True,
            fn_prob_threshold=0.8,
            fn_max_suppress_frac=1.0,
        )

        self.assertAlmostEqual(diag["alpha_n_scale_mean"], 0.85, places=5)
        self.assertAlmostEqual(diag["fn_selected_frac"], 1.0, places=5)

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


if __name__ == "__main__":
    unittest.main()

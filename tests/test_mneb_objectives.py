import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.objectives import (  # noqa: E402
    compute_branch_per_sample_contrastive_loss,
    compute_fnm_auxiliary_loss,
    compute_part_token_alignment,
    compute_part_token_score_matrix,
    compute_rde_auxiliary_loss,
)


class MNEBObjectivesTest(unittest.TestCase):
    def test_part_token_score_matrix_matches_alignment_loss(self):
        torch.manual_seed(2400)
        image_tokens = torch.randn(4, 4, 8, requires_grad=True)
        text_tokens = torch.randn(4, 6, 8, requires_grad=True)
        attention_mask = torch.ones(4, 6, dtype=torch.long)
        pids = torch.arange(4)

        score_mat, active = compute_part_token_score_matrix(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            num_parts=2,
            image_grid_hw=(2, 2),
        )
        score_loss = compute_branch_per_sample_contrastive_loss(
            score_mat[active][:, active],
            pids[active],
            temperature=0.07,
        ).mean()
        alignment_loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            temperature=0.07,
            image_grid_hw=(2, 2),
        )

        self.assertTrue(torch.isfinite(score_mat).all())
        self.assertTrue(torch.allclose(score_loss, alignment_loss, atol=1e-6))

    def test_fnm_aux_zero_when_no_candidates_is_grad_safe(self):
        image_features = torch.randn(4, 8, requires_grad=True)
        text_features = torch.randn(4, 8, requires_grad=True)
        pids = torch.arange(4)
        fn_mask = torch.zeros(4, 4, dtype=torch.bool)

        loss = compute_fnm_auxiliary_loss(
            image_features,
            text_features,
            pids,
            fn_candidate_mask=fn_mask,
        )
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(text_features.grad)

    def test_fnm_aux_is_finite_and_differentiable_with_candidates(self):
        torch.manual_seed(2401)
        image_features = torch.randn(4, 8, requires_grad=True)
        text_features = torch.randn(4, 8, requires_grad=True)
        pids = torch.arange(4)
        fn_mask = torch.zeros(4, 4, dtype=torch.bool)
        fn_mask[0, 1] = True
        fn_mask[2, 3] = True
        fn_prob = torch.zeros(4, 4)
        fn_prob[fn_mask] = 0.95

        loss = compute_fnm_auxiliary_loss(
            image_features,
            text_features,
            pids,
            fn_candidate_mask=fn_mask,
            fn_prob_matrix=fn_prob,
            theta_fn=0.8,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(text_features.grad)

    def test_rde_aux_zero_without_clean_anchors_is_grad_safe(self):
        global_score = torch.randn(4, 4, requires_grad=True)
        local_score = torch.randn(4, 4, requires_grad=True)
        pids = torch.arange(4)
        labels = torch.full((4,), -1, dtype=torch.long)

        loss = compute_rde_auxiliary_loss(global_score, local_score, pids, labels)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(global_score.grad)
        self.assertIsNotNone(local_score.grad)

    def test_rde_aux_is_finite_and_differentiable_with_clean_consensus(self):
        torch.manual_seed(2402)
        global_score = torch.randn(4, 4, requires_grad=True)
        local_score = torch.randn(4, 4, requires_grad=True)
        pids = torch.arange(4)
        labels = torch.ones(4, dtype=torch.long)

        loss = compute_rde_auxiliary_loss(
            global_score_mat=global_score,
            local_score_mat=local_score,
            pids=pids,
            consensus_labels=labels,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(global_score.grad)
        self.assertIsNotNone(local_score.grad)


if __name__ == "__main__":
    unittest.main()

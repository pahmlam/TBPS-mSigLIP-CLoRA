import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.objectives import compute_part_token_alignment  # noqa: E402


class PartTokenAlignmentLossTest(unittest.TestCase):
    def test_loss_is_finite_and_has_grad(self):
        torch.manual_seed(2400)
        image_tokens = torch.randn(4, 4, 8, requires_grad=True)
        text_tokens = torch.randn(4, 6, 8, requires_grad=True)
        attention_mask = torch.ones(4, 6, dtype=torch.long)
        pids = torch.arange(4)

        loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            image_grid_hw=(2, 2),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(image_tokens.grad)
        self.assertIsNotNone(text_tokens.grad)

    def test_matching_features_score_better_than_shuffled_text(self):
        batch_size = 4
        dim = 6
        base = torch.eye(batch_size, dim)
        image_tokens = base[:, None, :].repeat(1, 4, 1).requires_grad_(True)

        text_tokens = torch.zeros(batch_size, 6, dim)
        text_tokens[:, 1:5, :] = base[:, None, :]
        text_tokens = text_tokens.requires_grad_(True)

        attention_mask = torch.ones(batch_size, 6, dtype=torch.long)
        pids = torch.arange(batch_size)

        matched_loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            temperature=0.07,
            image_grid_hw=(2, 2),
        )
        shuffled_loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens.roll(shifts=1, dims=0),
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            temperature=0.07,
            image_grid_hw=(2, 2),
        )

        self.assertLess(matched_loss.item(), shuffled_loss.item())

    def test_all_invalid_text_tokens_returns_grad_safe_zero(self):
        image_tokens = torch.randn(3, 4, 8, requires_grad=True)
        text_tokens = torch.randn(3, 6, 8, requires_grad=True)
        attention_mask = torch.zeros(3, 6, dtype=torch.long)
        pids = torch.arange(3)

        loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            image_grid_hw=(2, 2),
        )
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(image_tokens.grad)
        self.assertIsNotNone(text_tokens.grad)

    def test_multi_positive_pids_are_supported(self):
        torch.manual_seed(2401)
        image_tokens = torch.randn(4, 4, 8, requires_grad=True)
        text_tokens = torch.randn(4, 6, 8, requires_grad=True)
        attention_mask = torch.ones(4, 6, dtype=torch.long)
        pids = torch.tensor([0, 0, 1, 1])

        loss = compute_part_token_alignment(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            pids=pids,
            num_parts=2,
            image_grid_hw=(2, 2),
        )

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

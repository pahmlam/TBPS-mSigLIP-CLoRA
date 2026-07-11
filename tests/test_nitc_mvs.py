import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.tbps import TBPS  # noqa: E402


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class FakeVision(nn.Module):
    def forward(self, images):
        pooled = images.flatten(1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=pooled[:, None, :])


class FakeText(nn.Module):
    def forward(self, input_ids, attention_mask):
        pooled = input_ids.float()
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=pooled[:, None, :])


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = FakeVision()
        self.text_model = FakeText()
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = torch.tensor(0.0)


def make_config(mode_marker="missing", circle=False):
    loss = AttrDict(
        {
            "NITC": True,
            "nitc_loss_weight": 1.0,
            "MVS": True,
            "CIR": circle,
            "CITC": False,
            "SS": False,
            "PART_ALIGN": False,
            "NACIR": False,
            "MNEB": False,
            "circle_loss_weight": 0.1,
            "circle_margin": 0.25,
            "circle_gamma": 128,
        }
    )
    if mode_marker != "missing":
        loss["nitc_mvs_mode"] = mode_marker
    return AttrDict(
        {
            "backbone": AttrDict({"embedding_dim": 2, "use_sigmoid": True}),
            "loss": loss,
        }
    )


def make_batch():
    return {
        "images": torch.ones(2, 1, 1, 2),
        "aug_images": torch.full((2, 1, 1, 2), 3.0),
        "caption_input_ids": torch.ones(2, 2, dtype=torch.long),
        "caption_attention_mask": torch.ones(2, 2, dtype=torch.long),
        "pids": torch.arange(2),
    }


class NITCMVSModeTest(unittest.TestCase):
    @staticmethod
    def _recording_nitc(calls):
        def fake_compute_constrative(**kwargs):
            image_features = kwargs["image_features"]
            calls.append(image_features.detach().clone())
            return image_features.mean()

        return fake_compute_constrative

    def test_missing_mode_preserves_baseline_nitc_behavior(self):
        calls = []
        model = TBPS(make_config(), FakeBackbone())

        with patch(
            "msiglip.model.tbps.objectives.compute_constrative",
            side_effect=self._recording_nitc(calls),
        ):
            ret = model(make_batch(), current_epoch=0)

        self.assertEqual(model.nitc_mvs_mode, "baseline")
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0], torch.ones_like(calls[0]))
        torch.testing.assert_close(calls[1], calls[0])
        self.assertAlmostEqual(ret["nitc_loss"].item(), 1.0)

    def test_direct_mode_uses_augmented_features_and_averages_losses(self):
        nitc_calls = []
        circle_calls = []
        model = TBPS(make_config("direct", circle=True), FakeBackbone())

        def fake_circle(**kwargs):
            image_features = kwargs["image_features"]
            circle_calls.append(image_features.detach().clone())
            return image_features.mean()

        with patch(
            "msiglip.model.tbps.objectives.compute_constrative",
            side_effect=self._recording_nitc(nitc_calls),
        ), patch(
            "msiglip.model.tbps.objectives.compute_cross_modal_circle",
            side_effect=fake_circle,
        ):
            ret = model(make_batch(), current_epoch=21)

        self.assertEqual(len(nitc_calls), 2)
        torch.testing.assert_close(nitc_calls[0], torch.ones_like(nitc_calls[0]))
        torch.testing.assert_close(nitc_calls[1], torch.full_like(nitc_calls[1], 3.0))
        self.assertAlmostEqual(ret["nitc_loss"].item(), 2.0)

        self.assertEqual(len(circle_calls), 2)
        torch.testing.assert_close(circle_calls[0], torch.ones_like(circle_calls[0]))
        torch.testing.assert_close(circle_calls[1], torch.full_like(circle_calls[1], 3.0))
        self.assertAlmostEqual(ret["circle_loss"].item(), 0.2)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nitc_mvs_mode"):
            TBPS(make_config("unknown"), FakeBackbone())


if __name__ == "__main__":
    unittest.main()

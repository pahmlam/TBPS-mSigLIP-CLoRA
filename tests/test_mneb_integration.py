import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(12, embed_dim)

    def forward(self, images):
        pooled = self.proj(images.flatten(1))
        tokens = pooled[:, None, :].repeat(1, 4, 1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=tokens)


class FakeText(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(32, embed_dim)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class FakeBackbone(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.vision_model = FakeVision(embed_dim)
        self.text_model = FakeText(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = torch.tensor(0.0)


def make_mneb_cfg(fnm_enabled=False, rde_enabled=False):
    return AttrDict(
        {
            "evidence_bank": AttrDict(
                {
                    "enabled": True,
                    "queue_size": 8,
                    "topk": 2,
                    "ema_beta": 0.9,
                    "loss_ema_alpha": 0.9,
                    "min_seen_epochs": 1,
                    "min_gmm_samples": 4,
                    "gmm_interval": 1,
                    "min_gmm_separation": 0.1,
                    "clean_threshold": 0.5,
                }
            ),
            "fnm_aux": AttrDict(
                {
                    "enabled": fnm_enabled,
                    "weight": 0.05,
                    "enable_epoch": 15,
                    "temperature": 0.07,
                    "margin": 0.25,
                    "theta_fn": 0.8,
                    "fn_prior": 0.003,
                    "max_candidate_frac": 0.02,
                    "min_pos_neg_gap": 0.0,
                    "local_agreement": True,
                }
            ),
            "rde_aux": AttrDict(
                {
                    "enabled": rde_enabled,
                    "weight": 0.05,
                    "enable_epoch": 15,
                    "margin": 0.2,
                    "temperature": 0.07,
                    "uncertain_policy": "no_op",
                }
            ),
        }
    )


def make_config(mneb=False, nacir=False):
    return AttrDict(
        {
            "img_size": (2, 2),
            "backbone": AttrDict(
                {
                    "embedding_dim": 8,
                    "use_sigmoid": False,
                    "vision_config": AttrDict({"patch_size": 1}),
                }
            ),
            "loss": AttrDict(
                {
                    "NITC": False,
                    "MVS": False,
                    "CIR": True,
                    "CITC": False,
                    "SS": False,
                    "PART_ALIGN": False,
                    "NACIR": nacir,
                    "MNEB": mneb,
                    "mneb_config": make_mneb_cfg(),
                    "circle_loss_weight": 0.1,
                    "circle_margin": 0.25,
                    "circle_gamma": 16,
                }
            ),
        }
    )


def make_batch(batch_size=4):
    return {
        "images": torch.randn(batch_size, 3, 2, 2),
        "caption_input_ids": torch.randint(0, 32, (batch_size, 6)),
        "caption_attention_mask": torch.ones(batch_size, 6, dtype=torch.long),
        "pids": torch.arange(batch_size),
        "id": torch.arange(batch_size),
    }


class MNEBIntegrationTest(unittest.TestCase):
    def test_mneb_false_does_not_instantiate_evidence_bank(self):
        model = TBPS(make_config(mneb=False), FakeBackbone(), num_train_samples=0)
        self.assertIsNone(model.evidence_bank)

    def test_mneb_true_aux_disabled_adds_diagnostics_only(self):
        torch.manual_seed(2400)
        model = TBPS(make_config(mneb=True), FakeBackbone(), num_train_samples=8)
        ret = model(make_batch(), current_epoch=10)

        self.assertIn("circle_loss", ret)
        self.assertIn("mneb_seen_frac", ret)
        self.assertNotIn("fnm_aux_loss", ret)
        self.assertNotIn("rde_aux_loss", ret)
        self.assertTrue(model.evidence_bank.sample_seen[:4].all())

    def test_mneb_and_nacir_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            TBPS(make_config(mneb=True, nacir=True), FakeBackbone(), num_train_samples=8)


if __name__ == "__main__":
    unittest.main()

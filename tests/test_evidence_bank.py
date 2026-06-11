import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.model.evidence_bank import EvidenceMemoryBank  # noqa: E402


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def make_cfg(**evidence_overrides):
    evidence_cfg = AttrDict(
        {
            "queue_size": 3,
            "topk": 2,
            "ema_beta": 0.5,
            "loss_ema_alpha": 0.5,
            "min_seen_epochs": 1,
            "min_gmm_samples": 4,
            "gmm_interval": 1,
            "min_gmm_separation": 0.1,
            "clean_threshold": 0.5,
        }
    )
    evidence_cfg.update(evidence_overrides)
    return AttrDict(
        {
            "evidence_bank": evidence_cfg,
            "fnm_aux": AttrDict({"fn_prior": 0.003, "enable_epoch": 1}),
            "rde_aux": AttrDict({"enable_epoch": 1}),
        }
    )


class EvidenceMemoryBankTest(unittest.TestCase):
    def test_buffer_initialization_and_device_movement(self):
        bank = EvidenceMemoryBank(num_train_samples=5, embed_dim=4, config=make_cfg())
        self.assertEqual(bank.image_global.shape, (5, 4))
        self.assertEqual(bank.text_global.shape, (5, 4))
        self.assertEqual(bank.queue_ids.shape, (3,))
        bank = bank.to(torch.device("cpu"))
        self.assertEqual(bank.image_global.device.type, "cpu")

    def test_queue_bounds_and_loss_ema_update(self):
        torch.manual_seed(2400)
        bank = EvidenceMemoryBank(num_train_samples=5, embed_dim=4, config=make_cfg())
        ids = torch.tensor([0, 1, 2])
        feats = torch.randn(3, 4)
        pids = torch.arange(3)
        losses = torch.tensor([1.0, 2.0, 3.0])

        bank.update_batch(ids, feats, feats, pids, global_losses=losses)
        self.assertTrue(torch.equal(bank.global_loss_ema[:3], losses))

        bank.update_batch(
            torch.tensor([0, 3, 4]),
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.arange(3),
            global_losses=torch.tensor([3.0, 4.0, 5.0]),
        )
        self.assertAlmostEqual(bank.global_loss_ema[0].item(), 2.0, places=5)
        self.assertLessEqual(bank.queue_count.item(), bank.queue_size)

    def test_gmm_fallback_before_enough_samples(self):
        bank = EvidenceMemoryBank(
            num_train_samples=5,
            embed_dim=4,
            config=make_cfg(min_gmm_samples=10),
        )
        bank.update_batch(
            torch.tensor([0, 1]),
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.arange(2),
            global_losses=torch.tensor([1.0, 2.0]),
        )
        diag = bank.refit_epoch(0)
        self.assertEqual(diag["global_fallback"], 1.0)
        self.assertTrue(torch.all(bank.global_clean_prob == 1.0))
        self.assertTrue(torch.all(bank.consensus_label == -1))

    def test_consensus_labels_after_global_local_gmm(self):
        torch.manual_seed(2401)
        bank = EvidenceMemoryBank(num_train_samples=8, embed_dim=4, config=make_cfg())
        ids = torch.arange(8)
        feats = torch.randn(8, 4)
        pids = torch.arange(8)
        losses = torch.tensor([0.10, 0.11, 0.12, 0.13, 4.0, 4.1, 4.2, 4.3])

        bank.update_batch(
            ids,
            feats,
            feats,
            pids,
            global_losses=losses,
            local_losses=losses,
            local_image_features=feats,
            local_text_features=feats,
        )
        bank.refit_epoch(0)

        self.assertTrue((bank.consensus_label[:4] == 1).any())
        self.assertTrue((bank.consensus_label[4:] == 0).any())


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft import LoraConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LORA_DIR = PROJECT_ROOT / "configs" / "lora"


class LoraConfigVariantsTest(unittest.TestCase):
    def _load_config(self, name):
        cfg = OmegaConf.load(LORA_DIR / f"{name}.yaml")
        return instantiate(cfg)

    def test_all_new_lora_configs_instantiate(self):
        names = [
            "attn_ffn_r32",
            "attn_ffn_r64",
            "attn_ffn_r32_rslora",
            "attn_ffn_r32_pissa",
            "attn_ffn_r32_dora",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertIsInstance(self._load_config(name), LoraConfig)

    def test_target_modules_include_attention_and_ffn(self):
        cfg = self._load_config("attn_ffn_r32")
        target_modules = set(cfg.target_modules)

        self.assertTrue({"q_proj", "k_proj", "v_proj", "out_proj"}.issubset(target_modules))
        self.assertTrue({"fc1", "fc2"}.issubset(target_modules))

    def test_variant_flags_are_accepted_by_local_peft(self):
        rslora = self._load_config("attn_ffn_r32_rslora")
        pissa = self._load_config("attn_ffn_r32_pissa")
        dora = self._load_config("attn_ffn_r32_dora")

        self.assertTrue(rslora.use_rslora)
        self.assertEqual(pissa.init_lora_weights, "pissa_niter_4")
        self.assertTrue(dora.use_dora)


if __name__ == "__main__":
    unittest.main()

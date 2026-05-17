from pathlib import Path

import fire
from omegaconf import OmegaConf, DictConfig
import lightning as L

from msiglip.lightning_data import TBPSDataModule
from msiglip.lightning_models import LitTBPS
from lightning.pytorch import seed_everything


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)
OmegaConf.register_new_resolver("eval", eval)


def load_test_loader(dataset_name: str, config: DictConfig):
    """
    Load the dataset from the configuration.

    Args:
        dataset_name (str): The name of the dataset.
        config (DictConfig): The configuration.
    Returns:
        test_loader (DataLoader): The test loader.
    """
    config.dataset.dataset_name = dataset_name
    dm = TBPSDataModule(config)
    dm.setup()
    test_loader = dm.test_dataloader()

    return test_loader


def run_test(ckpt_path: str | Path, dataset_name: str):
    model = LitTBPS.load_from_checkpoint(ckpt_path)
    config = model.hparams.config
    seed_everything(config.seed)
    test_loader = load_test_loader(dataset_name, config)

    trainer = L.Trainer(**config.trainer)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    fire.Fire(run_test)

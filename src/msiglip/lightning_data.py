import os
import warnings
from random import Random

import numpy as np
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader
from loguru import logger

from msiglip.data.augmentation.transform import build_image_aug_pool, build_text_aug_pool
from msiglip.data.bases import (
    ImageDataset,
    ImageTextDataset,
    TextDataset,
    inject_noisy_correspondence,
)
from msiglip.data.cuhkpedes import CUHKPEDES
from msiglip.data.icfgpedes import ICFGPEDES
from msiglip.data.rstpreid import RSTPReid
from msiglip.data.sampler import RandomIdentitySampler

# from msiglip.data.sampler_ddp import RandomIdentitySampler_DDP
from msiglip.data.vn3k_mixed import VN3K_MIXED
from msiglip.data.vn3k_vi import VN3K_VI
from msiglip.data.vn3k_en import VN3K_EN
from msiglip.data.cuhk_10_percent_vn3k_mix import TenPercentCUHK_VN3KMIX
from msiglip.data.prw_tps_cn import PRW_TPS_CN

# from msiglip.utils.comm import get_world_size
from msiglip.utils.tokenizer_utils import get_tokenizer

# from torch.utils.data.distributed import DistributedSampler
# Filter UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class TBPSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        __factory = {
            "CUHK-PEDES": CUHKPEDES,
            "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid,
            "VN3K_EN": VN3K_EN,
            "VN3K_VI": VN3K_VI,
            "VN3K_MIXED": VN3K_MIXED,
            "TEN_PERCENT_CUHK_VN3K_MIX": TenPercentCUHK_VN3KMIX,
            "PRW_TPS_CN": PRW_TPS_CN,
        }
        self.config = config
        print(config.dataset.dataset_name)
        self.dataset = __factory[config.dataset.dataset_name](
            root=config.dataset_root_dir, seed=config.seed
        )
        self.tokenizer = get_tokenizer(config.tokenizer)
        self.num_classes = len(self.dataset.train_id_container)
        # Set up transforms
        self._prepare_augmentation_pool()

    def _prepare_augmentation_pool(self):
        self.image_aug_pool = build_image_aug_pool(self.config.aug.img.augment_cfg)
        self.text_aug_pool = build_text_aug_pool(self.config.aug.text.augment_cfg)
        self.image_random_k = self.config.aug.image_random_k
        self.text_random_k = self.config.aug.text_random_k
        if self.config.loss.SS:
            self.ss_aug = True
        else:
            self.ss_aug = None
        self.mean = self.config.aug.img.mean
        self.std = self.config.aug.img.std

    def setup(self, stage=None):
        if self.config.dataset.proportion:
            if self.config.dataset.dataset_name == "TEN_PERCENT_CUHK_VN3K_MIX":
                raise NotImplementedError(
                    "This mixed dataset does not support subset sampling"
                )
            # TODO: The subset sample has to contain a specific number of PID
            # Shuffle dataset once for reproducible folds
            shuffled_train_data = self.dataset.train[:]
            Random(self.config.seed).shuffle(shuffled_train_data)

            proportion = self.config.dataset.proportion
            num_folds = int(1 / proportion)
            fold_size = int(len(shuffled_train_data) * proportion)

            # Get current fold index from config.
            # You need to add `fold_id` to your dataset config and vary it for each run.
            fold_id = self.config.dataset.get("fold_id", 0)

            if not (0 <= fold_id < num_folds):
                raise ValueError(
                    f"fold_id {fold_id} is out of bounds for {num_folds} folds. "
                    f"Valid fold_id is from 0 to {num_folds - 1}."
                )

            # Calculate start and end index for the fold
            start_idx = fold_id * fold_size
            end_idx = start_idx + fold_size

            # Select the fold. Slicing handles the case where the last fold might be smaller.
            self.dataset.train = shuffled_train_data[start_idx:end_idx]

            logger.info(
                f"Using fold {fold_id}/{num_folds - 1} of the training set, with {len(self.dataset.train)} samples."
            )

        # --- Noisy correspondence injection ---
        noisy_rate = self.config.dataset.get("noisy_rate", 0.0)
        if noisy_rate > 0:
            noisy_file = self.config.dataset.get("noisy_file", None)
            if noisy_file is None:
                artifacts_root = self.config.get("paths", {}).get(
                    "artifacts_root", "artifacts"
                )
                noiseindex_dir = os.path.join(
                    get_original_cwd(), artifacts_root, "training", "noiseindex"
                )
                os.makedirs(noiseindex_dir, exist_ok=True)
                noisy_file = os.path.join(
                    noiseindex_dir,
                    f"{self.config.dataset.dataset_name}_{noisy_rate}.npy",
                )

            self.dataset.original_train = list(self.dataset.train)
            self.dataset.train, self.real_correspondences = inject_noisy_correspondence(
                dataset=self.dataset.train,
                noisy_rate=noisy_rate,
                noisy_file=noisy_file,
            )

        if stage == "fit" or stage is None:
            self.train_set = ImageTextDataset(
                dataset=self.dataset.train,
                tokenizer=self.tokenizer,
                ss_aug=self.ss_aug,
                image_augmentation_pool=self.image_aug_pool,
                text_augmentation_pool=self.text_aug_pool,
                image_random_k=self.image_random_k,
                text_random_k=self.text_random_k,
                truncate=True,
                image_size=self.config.aug.img.size,
                is_train=True,
                mean=self.mean,
                std=self.std,
            )

            if noisy_rate > 0:
                self.train_set.real_correspondences = self.real_correspondences
                self.train_set.original_dataset = self.dataset.original_train

            logger.info("Validation set is available")

            self.val_img_set = ImageDataset(
                dataset=self.dataset.val,
                is_train=False,
                image_size=self.config.aug.img.size,
                mean=self.mean,
                std=self.std,
            )
            self.val_txt_set = TextDataset(
                dataset=self.dataset.val,
                tokenizer=self.tokenizer,
                is_train=False,
            )

        if stage == "test" or stage is None:
            self.test_img_set = ImageDataset(
                dataset=self.dataset.test,
                image_size=self.config.aug.img.size,
                is_train=False,
                mean=self.mean,
                std=self.std,
            )
            self.test_txt_set = TextDataset(
                dataset=self.dataset.test,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        if self.config.dataset.sampler == "identity":
            if self.config.distributed:
                raise NotImplementedError(
                    "Distributed sampler is not implemented yet, please use random sampler"
                )
                logger.info("using ddp random identity sampler")
                logger.info("DISTRIBUTED TRAIN START")
                # mini_batch_size = self.config.dataset.batch_size // get_world_size()
                # data_sampler = RandomIdentitySampler_DDP(
                #     self.dataset.train,
                #     self.config.dataset.batch_size,
                #     self.config.dataset.num_instance,
                # )
                # batch_sampler = torch.utils.data.sampler.BatchSampler(
                #     data_sampler, mini_batch_size, True
                # )
                # return DataLoader(
                #     self.train_set,
                #     batch_sampler=batch_sampler,
                #     num_workers=self.config.dataset.num_workers,
                #     # collate_fn=self.collate_fn,
                #     drop_last=False,
                # )
            else:
                logger.info(
                    f"using random identity sampler: batch_size: {self.config.dataset.batch_size}, id: {self.config.dataset.batch_size // self.config.dataset.num_instance}, instance: {self.config.dataset.num_instance}"
                )
                return DataLoader(
                    self.train_set,
                    batch_size=self.config.dataset.batch_size,
                    sampler=RandomIdentitySampler(
                        self.dataset.train,
                        self.config.dataset.batch_size,
                        self.config.dataset.num_instance,
                    ),
                    num_workers=self.config.dataset.num_workers,
                    # collate_fn=self.collate_fn,
                    drop_last=False,
                )
        elif self.config.dataset.sampler == "random":
            logger.info("using random sampler")
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset.batch_size,
                shuffle=True,
                num_workers=self.config.dataset.num_workers,
                # collate_fn=self.collate_fn,
                drop_last=True,
            )
        else:
            logger.error(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    self.config.dataset.sampler
                )
            )
            raise ValueError("Unsupported sampler type")

    def val_dataloader(self):
        val_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        val_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_val = {
            "img": val_img_loader,
            "txt": val_txt_loader,
        }
        combined_loader = CombinedLoader(combined_val, mode="max_size")
        return combined_loader

    def test_dataloader(self):
        test_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        test_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_test = {
            "img": test_img_loader,
            "txt": test_txt_loader,
        }
        combined_loader = CombinedLoader(combined_test, mode="sequential")
        return combined_loader
    def tsne_dataloader(self):
        """
        Deterministic dataloader for visualization (t-SNE).
        - No RandomIdentitySampler
        - No CombinedLoader
        - Same ordering every run
        """
        dataset = ImageTextDataset(
            dataset=self.dataset.test,
            tokenizer=self.tokenizer,
            ss_aug=None,
            image_augmentation_pool=None,
            text_augmentation_pool=None,
            image_random_k=1,
            text_random_k=1,
            truncate=True,
            image_size=self.config.aug.img.size,
            is_train=False,
            mean=self.mean,
            std=self.std,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,          # 🔴 QUAN TRỌNG
            num_workers=self.config.dataset.num_workers,
            drop_last=False,
        )


import logging
import os
import random
from typing import Tuple, Optional, Callable, List

import numpy as np
import torch
from loguru import logger
from prettytable import PrettyTable
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .augmentation.transform import (
    get_self_supervised_augmentation,
    get_image_transform,
    get_text_transform,
)
from msiglip.utils.iotools import read_image


def inject_noisy_correspondence(
    dataset: list[tuple],
    noisy_rate: float,
    noisy_file: str | None = None,
) -> tuple[list[tuple], np.ndarray]:
    """Inject noisy correspondence by shuffling captions at the given rate.

    For each sample selected as noisy, its caption is replaced with a caption
    from another noisy sample. The pid stays with the image, creating mismatched
    (image, caption) pairs that the model treats as positives.

    Args:
        dataset: list of (pid, image_id, img_path, caption) tuples (mutated in-place).
        noisy_rate: fraction of samples to corrupt (0.0 = clean, 1.0 = all noisy).
        noisy_file: path to .npy for persisting/loading the noise index for reproducibility.

    Returns:
        The mutated dataset and a boolean array (1=clean, 0=noisy).
    """
    nums = len(dataset)
    captions = [i[3] for i in dataset]
    images = [i[2] for i in dataset]
    image_ids = [i[1] for i in dataset]
    pids = [i[0] for i in dataset]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        random.seed(123)
        if noisy_file and os.path.exists(noisy_file):
            logger.info(f"Loading noisy index from {noisy_file}")
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            if noisy_file:
                os.makedirs(os.path.dirname(noisy_file), exist_ok=True)
                np.save(noisy_file, noisy_inx)

    real_correspondences = np.ones(nums, dtype=np.int32)
    for i in range(nums):
        if noisy_inx[i] != i:
            real_correspondences[i] = 0
        dataset[i] = (pids[i], image_ids[i], images[i], captions[noisy_inx[i]])

    num_clean = int(np.sum(real_correspondences))
    num_noisy = nums - num_clean
    logger.info(
        f"Noisy correspondence: rate={noisy_rate}, clean={num_clean}, "
        f"noisy={num_noisy}, total={nums}"
    )

    return dataset, real_correspondences


def inject_false_negative_labels(
    dataset: list[tuple],
    fn_noisy_rate: float,
    fn_noisy_file: str | None = None,
) -> tuple[list[tuple], np.ndarray]:
    """Inject synthetic false negatives by splitting train labels.

    For selected identities with at least two samples, half of their samples are
    assigned a new fake PID. The image, image_id, and caption stay unchanged.
    This makes same-person samples become labeled negatives during training.

    Args:
        dataset: list of (pid, image_id, img_path, caption) tuples (mutated in-place).
        fn_noisy_rate: target fraction of samples whose PID should be changed.
        fn_noisy_file: path to .npy storing the full PID mapping for reproducibility.

    Returns:
        The mutated dataset and a boolean array (1=PID changed, 0=unchanged).
    """
    nums = len(dataset)
    if nums == 0:
        return dataset, np.zeros(0, dtype=np.int32)
    if fn_noisy_rate <= 0:
        logger.info(
            f"False-negative label noise: rate={fn_noisy_rate}, changed=0, "
            f"unchanged={nums}, total={nums}"
        )
        return dataset, np.zeros(nums, dtype=np.int32)

    original_pids = np.asarray([sample[0] for sample in dataset], dtype=np.int64)
    image_ids = [sample[1] for sample in dataset]
    images = [sample[2] for sample in dataset]
    captions = [sample[3] for sample in dataset]

    if fn_noisy_file and os.path.exists(fn_noisy_file):
        logger.info(f"Loading false-negative PID mapping from {fn_noisy_file}")
        mapped_pids = np.load(fn_noisy_file).astype(np.int64)
        if len(mapped_pids) != nums:
            raise ValueError(
                f"False-negative PID mapping length mismatch: got {len(mapped_pids)}, expected {nums}"
            )
    else:
        mapped_pids = original_pids.copy()
        if fn_noisy_rate > 0:
            rng = np.random.default_rng(123)
            target_changes = int(fn_noisy_rate * nums)
            next_fake_pid = int(original_pids.max()) + 1

            pid_to_indices: dict[int, list[int]] = {}
            for idx, pid in enumerate(original_pids):
                pid_to_indices.setdefault(int(pid), []).append(idx)

            eligible_pids = [pid for pid, indices in pid_to_indices.items() if len(indices) >= 2]
            rng.shuffle(eligible_pids)

            num_changed = 0
            for pid in eligible_pids:
                if num_changed >= target_changes:
                    break

                indices = np.asarray(pid_to_indices[pid], dtype=np.int64)
                rng.shuffle(indices)
                split_size = max(1, len(indices) // 2)
                remaining_budget = target_changes - num_changed
                split_size = min(split_size, remaining_budget)
                if split_size <= 0:
                    break

                selected = indices[:split_size]
                mapped_pids[selected] = next_fake_pid
                next_fake_pid += 1
                num_changed += split_size

        if fn_noisy_file:
            dirname = os.path.dirname(fn_noisy_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            np.save(fn_noisy_file, mapped_pids)

    changed = (mapped_pids != original_pids).astype(np.int32)
    for i in range(nums):
        dataset[i] = (int(mapped_pids[i]), image_ids[i], images[i], captions[i])

    num_changed = int(np.sum(changed))
    logger.info(
        f"False-negative label noise: rate={fn_noisy_rate}, changed={num_changed}, "
        f"unchanged={nums - num_changed}, total={nums}"
    )

    return dataset, changed


class BaseDataset(object):
    """
    Base class for all datasets before loaded into DataLoader
    to show dataset statistics and information.

    Args:
        root: str, root directory of the dataset

    """

    logger = logger

    def show_dataset_info(self):
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(["subset", "ids", "images", "captions"])
        if self.train:
            num_train_pids, num_train_imgs, num_train_captions = (
                len(self.train_id_container),
                len(self.train_annos),
                len(self.train),
            )
            table.add_row(["train", num_train_pids, num_train_imgs, num_train_captions])
        if self.test:
            num_test_pids, num_test_imgs, num_test_captions = (
                len(self.test_id_container),
                len(self.test_annos),
                len(self.test["captions"]),
            )
            table.add_row(["test", num_test_pids, num_test_imgs, num_test_captions])
        if self.val:
            num_val_pids, num_val_imgs, num_val_captions = (
                len(self.val_id_container),
                len(self.val_annos),
                len(self.val["captions"]),
            )
            table.add_row(["val", num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info("\n" + str(table))


class PreloadedDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        ss_aug: bool = False,
        image_augmentation_pool: List[Callable] = None,
        text_augmentation_pool: List[Callable] = None,
        image_random_k: int = 2,
        text_random_k: int = 2,
        truncate: bool = True,
        image_size: Tuple[int, int] = (384, 128),
        is_train: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Base class for all datasets that are preloaded into memory
        before loaded into DataLoader for training.

        Args:
            dataset: list, list of data samples
            tokenizer: Optional[PreTrainedTokenizer], tokenizer object
            ss_aug: bool, whether to apply self-supervised augmentation
            image_augmentation_pool: List[Callable], list of image augmentation functions
            text_augmentation_pool: List[Callable], list of text augmentation functions
            image_random_k: int, number of augmentation functions to apply for image
            text_random_k: int, number of augmentation functions to apply for text
            -1 means apply all augmentation functions
            truncate: bool, truncate text to max length
            image_size: Tuple[int, int], image size
            is_train: bool, whether the dataset is for training
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.ss_aug = ss_aug
        self.image_augmentation_pool = image_augmentation_pool
        self.text_augmentation_pool = text_augmentation_pool
        self.image_random_k = image_random_k
        self.text_random_k = text_random_k
        self.truncate = truncate
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

        if self.ss_aug:
            self.ss_image_transform = get_self_supervised_augmentation(
                self.image_size, self.mean, self.std
            )

    def __len__(self):
        return len(self.dataset)

    def apply_image_transform(self, img):
        self.current_img_transforms = get_image_transform(
            aug_pool=self.image_augmentation_pool,
            size=self.image_size,
            mean=self.mean,
            std=self.std,
            k=self.image_random_k,
            is_train=self.is_train,
        )
        augmented_img = self.current_img_transforms(img)
        return augmented_img

    def apply_text_transform(self, caption):
        self.current_text_transforms = get_text_transform(
            aug_pool=self.text_augmentation_pool,
            k=self.text_random_k,
            is_train=self.is_train,
        )
        caption = self.current_text_transforms(caption)
        return caption

    def apply_ss_image_transform(self, img):
        augmented_img = self.ss_image_transform(img)
        return augmented_img

    def tokenize(
        self,
        caption: str,
        padding: str = "max_length",
        return_tensors: str = "pt",
        return_attention_mask: bool = True,
    ):
        truncation = self.truncate
        inputs = self.tokenizer(
            caption,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


class ImageTextDataset(PreloadedDataset):
    def __init__(self, *args, **kwargs):
        """
        Dataset containing image and text samples.

        Args:
            dataset: list, list of data samples
            tokenizer: Optional[PreTrainedTokenizer], tokenizer object
            ss_aug: bool, whether to apply self-supervised augmentation
            image_augmentation_pool: List[Callable], list of image augmentation functions
            text_augmentation_pool: List[Callable], list of text augmentation functions
            image_random_k: int, number of augmentation functions to apply for image
            text_random_k: int, number of augmentation functions to apply for text
            -1 means apply all augmentation functions
            truncate: bool, truncate text to max length
            image_size: Tuple[int, int], image size
            is_train: bool, whether the dataset is for training
        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        try:
            original_img = read_image(img_path)
        except IOError:
            return self.__getitem__((index + 1) % len(self.dataset))

        # Get two augmentations of the same image for contrastive learning
        img = self.apply_image_transform(original_img)
        aug = self.apply_image_transform(original_img)

        caption = self.apply_text_transform(caption)

        tokens = self.tokenize(
            caption,
        )

        ret = {
            "id": index,
            "pids": pid,
            "image_ids": image_id,
            "images": img,
            "aug_images": aug,
        }

        for key, value in tokens.items():
            ret[f"caption_{key}"] = value

        if self.ss_aug:
            # Apply self-supervised augmentation if available
            ss_aug1 = self.apply_ss_image_transform(original_img)
            ss_aug2 = self.apply_ss_image_transform(original_img)

            ret.update(
                {
                    "ss_images1": ss_aug1,
                    "ss_images2": ss_aug2,
                }
            )

        return ret


class ImageDataset(PreloadedDataset):
    def __init__(self, *args, **kwargs):
        """
        Dataset containing image samples.

        Args:
            dataset: list, list of data samples
            tokenizer: Optional[PreTrainedTokenizer], tokenizer object
            ss_aug: bool, whether to apply self-supervised augmentation
            image_augmentation_pool: List[Callable], list of image augmentation functions
            text_augmentation_pool: List[Callable], list of text augmentation functions
            image_random_k: int, number of augmentation functions to apply for image
            text_random_k: int, number of augmentation functions to apply for text
            -1 means apply all augmentation functions
            truncate: bool, truncate text to max length
            image_size: Tuple[int, int], image size
            is_train: bool, whether the dataset is for training
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.dataset["image_pids"])

    def __getitem__(self, index):
        image_pids, img_path = (
            self.dataset["image_pids"][index],
            self.dataset["img_paths"][index],
        )
        original_img = read_image(img_path)

        # Image Transform here only includes: Resize, ToTensor, Normalize
        img = self.apply_image_transform(original_img)

        return {
            "pids": image_pids,
            "images": img,
        }


class TextDataset(PreloadedDataset):
    def __init__(self, *args, **kwargs):
        """
        Dataset containing text samples.

        Args:
            dataset: list, list of data samples
            tokenizer: Optional[PreTrainedTokenizer], tokenizer object
            ss_aug: bool, whether to apply self-supervised augmentation
            image_augmentation_pool: List[Callable], list of image augmentation functions
            text_augmentation_pool: List[Callable], list of text augmentation functions
            image_random_k: int, number of augmentation functions to apply for image
            text_random_k: int, number of augmentation functions to apply for text
            -1 means apply all augmentation functions
            truncate: bool, truncate text to max length
            image_size: Tuple[int, int], image size
            is_train: bool, whether the dataset is for training
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.dataset["caption_pids"])

    def __getitem__(self, index):
        caption_pids, caption = (
            self.dataset["caption_pids"][index],
            self.dataset["captions"][index],
        )
        # There is no text transformation here
        caption = self.apply_text_transform(caption)

        tokens = self.tokenize(
            caption,
        )

        ret = {
            "pids": caption_pids,
            "text": caption,
        }

        for key, value in tokens.items():
            ret[f"caption_{key}"] = value

        return ret


class ImageTextMLMDataset(PreloadedDataset):
    def __init__(self, mlm_probability: float = 0.15, *args, **kwargs):
        """
        Dataset containing image and text samples for masked language modeling.

        Args:
            dataset: list, list of data samples
            tokenizer: Optional[PreTrainedTokenizer], tokenizer object
            ss_aug: bool, whether to apply self-supervised augmentation
            image_augmentation_pool: List[Callable], list of image augmentation functions
            text_augmentation_pool: List[Callable], list of text augmentation functions
            image_random_k: int, number of augmentation functions to apply for image
            text_random_k: int, number of augmentation functions to apply for text
            -1 means apply all augmentation functions
            truncate: bool, truncate text to max length
            image_size: Tuple[int, int], image size
            is_train: bool, whether the dataset is for training
        """
        super().__init__(*args, **kwargs)
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        original_img = read_image(img_path)

        img = self.apply_image_transform(original_img)
        aug = self.apply_image_transform(original_img)
        caption = self.apply_text_transform(caption)

        tokens = self.tokenize(
            caption,
        )

        inputs, labels = self.mask_tokens(
            inputs=tokens,
            mask_token=self.tokenizer.mask_token_id,
            special_tokens=self.tokenizer.special_token_ids,
        )

        ret = {
            "id": index,
            "pids": pid,
            "image_ids": image_id,
            "images": img,
            "mlm_labels": labels,
            "aug_images": aug,
        }

        for key, value in tokens.items():
            ret[f"caption_{key}"] = value
        for key, value in inputs.items():
            ret[f"mlm_{key}"] = value

        if self.ss_aug:
            # Apply self-supervised augmentation if available
            ss_aug1 = self.apply_ss_image_transform(original_img)
            ss_aug2 = self.apply_ss_image_transform(original_img)

            ret.update(
                {
                    "ss_images1": ss_aug1,
                    "ss_images2": ss_aug2,
                }
            )

        return ret

    def mask_tokens(
        self,
        inputs: dict,
        mask_token,
        special_tokens=None,
        mlm_probability=0.15,
        special_tokens_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original.

        Args:
            inputs: dict, input tokens
            mask_token: int, mask token id
            special_tokens: list, special token ids
            mlm_probability: float, probability of masking
            special_tokens_mask: list, special token mask

        Return:
            inputs: dict, masked input tokens
            labels: tensor, masked labels
        """
        labels = inputs["input_ids"].clone()

        new_inputs = {k: v.clone() for k, v in inputs.items()}
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(
            labels.shape, mlm_probability
        )  # Create a matrix of the same shape as labels filled with mlm_probability
        if special_tokens_mask is None:
            special_tokens_mask = [
                1 if val in special_tokens else 0 for val in labels.tolist()
            ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        # if tokenizer._pad_token is not None:
        #     padding_mask = labels.eq(tokenizer.pad_token_id)
        #     probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = (
            self.tokenizer.pad_token_id
        )  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        new_inputs["input_ids"][indices_replaced] = mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.tokenizer.true_vocab_size, labels.shape, dtype=torch.long
        )
        new_inputs["input_ids"][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return new_inputs, labels

"""
Image transformation functions.
"""

import logging
import random
from typing import Callable, Tuple
from functools import reduce

import numpy as np
from PIL import ImageFilter
from torchvision import transforms

from msiglip.utils.parse_module_str import parse_module_str

logger = logging.getLogger(__name__)


def get_self_supervised_augmentation(img_size, mean, std):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[0.1, 2.0]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(mean=mean, std=std)

    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), antialias=True),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return aug


def get_image_transform(
    aug_pool,
    size: Tuple[int, int] = (384, 128),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    k: int = 2,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Build a pool of image augmentation functions.

    Args:
        augment_cfg (dict): A dictionary of image augmentation functions.
        size: The size of the output image.
        k: The number of augmentation functions to apply.
        is_train: Whether the augmentation is for training.
        If is not for training, only resize and normalize will be applied.
    Returns:
        A callable that applies a random selection of image augmentation functions.
    """
    EXCLUDING = [
        "Resize",
        "ToTensor",
        "Normalize",
        "PILResize",
        "Rescale",
    ]
    if not is_train or not aug_pool:
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(), normalize]
        )
        return transform

    if k == -1:
        return transforms.Compose(aug_pool)

    # Randomly select k augmentations from the pool excluding Resize, ToTensor, Normalize, PILResize, Rescale
    random_pool = [aug for aug in aug_pool if aug.__class__.__name__ not in EXCLUDING]
    if k > len(random_pool):
        k = len(random_pool)
    aug_choice = np.random.choice(random_pool, k)

    # Combine selected augmentations with Resize, ToTensor, Normalize, retaining the order
    image_aug = [
        aug
        for aug in aug_pool
        if aug in aug_choice or aug.__class__.__name__ in EXCLUDING
    ]

    transform = transforms.Compose(image_aug)
    return transform


def build_image_aug_pool(augment_cfg: dict = None):
    additional_aug = []
    if augment_cfg:
        for aug_type, aug_params in augment_cfg.items():
            try:
                if aug_params:
                    aug = parse_module_str(aug_type)(**aug_params)
                else:
                    aug = parse_module_str(aug_type)()
            except TypeError as e:
                raise TypeError(f"Augmentation {aug_type} requires parameters: {e}")
            additional_aug.append(aug)
        logger.info(f"Using image augmentation: {additional_aug} for training.")
        return additional_aug

    return None


def get_text_transform(
    aug_pool,
    k: int = 1,
    is_train: bool = True,
) -> Callable:
    """
    Build a pool of text augmentation functions.

    Args:
        aug_pool: A list of text augmentation functions.
        k: The number of augmentation functions to apply.
        is_train: Whether the augmentation is for training.
        If not for training, no augmentation will be applied.
    Returns:
        A callable that applies a random selection of text augmentation functions.
    """
    text_aug = []
    if is_train and aug_pool:
        if k == -1:
            k = len(aug_pool)

        aug_choice = np.random.choice(aug_pool, k)
        text_aug.extend(aug_choice)

    if text_aug:
        return lambda x: reduce(lambda x, f: f(x), text_aug, x)
    else:
        return lambda x: x  # Identity function if no augmentations


def build_text_aug_pool(augment_cfg: dict = None):
    if augment_cfg:
        additional_aug = []
        for aug_type, aug_params in augment_cfg.items():
            aug = parse_module_str(aug_type)(**aug_params)
            additional_aug.append(aug)
        logger.info(f"Using text augmentation: {additional_aug} for training.")
        return additional_aug

    return None

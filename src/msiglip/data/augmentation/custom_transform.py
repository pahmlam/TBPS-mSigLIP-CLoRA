from typing import Tuple

import torch
from PIL import Image


class PILResize(torch.nn.Module):
    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        resample: int = 3,
        reducing_gap: None = None,
    ):
        """
        Args:
            size (tuple of int): The target size of the output image (width, height).
            resample (int): An optional resampling filter. This can be one of
                :attr:`PIL.Image.NEAREST`, :attr:`PIL.Image.BOX`, :attr:`PIL.Image.BILINEAR`,
                :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC` or :attr:`PIL.Image.LANCZOS`.
                Default is :attr:`PIL.Image.BICUBIC`.
            reducing_gap (int): An optional resampling filter. Default is None.
        """
        super().__init__()
        # reverse the size
        self.size = size[::-1]
        self.resample = resample
        self.reducing_gap = reducing_gap

    def forward(self, x: Image.Image):
        return x.resize(self.size, self.resample, self.reducing_gap)


class Rescale(torch.nn.Module):
    def __init__(self, rescale_factor: float):
        super().__init__()
        self.rescale_factor = rescale_factor

    def forward(self, x: torch.tensor):
        return x * self.rescale_factor

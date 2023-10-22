import cv2 as cv
from typing import Tuple, Union

import numpy as np

from common.models import Image


class Mask(Image):
    NERVE_MASK_COLOR = (1, 1, 1)  # Without tumor
    NERVE_BOX_MASK_COLOR = (11, 11, 11)  # Without tumor, boxes around NERVE_MASK_COLOR
    PNI_MASK_COLOR = (2, 2, 2)  # Perineural invasion boundaries (curves)
    TUMOR_MASK_COLOR = (13, 13, 13)  # Without nerves
    EMPTY_MASK_COLOR = (14, 14, 14)  # Non-tumor without nerves

    def __init__(self, image: Union[np.ndarray, str]):
        super().__init__(image)
        self._contains_nerve = None
        self._contains_tumor = None
        self._contains_nontumor_without_nerve = None
        self._contains_pni = None

    def contains_nerve(self, threshold: float = 0.1) -> bool:
        if self._contains_nerve is not None:
            return self._contains_nerve
        nerve_pixels = self.count_mask_pixels(self.NERVE_MASK_COLOR)
        self._contains_nerve = nerve_pixels / (self.width * self.height) > threshold
        return self._contains_nerve

    def contains_tumor(self, threshold: float = 0.7) -> bool:
        if self._contains_tumor is not None:
            return self._contains_tumor
        tumor_pixels = self.count_mask_pixels(self.TUMOR_MASK_COLOR)
        self._contains_tumor = tumor_pixels / (self.width * self.height) > threshold
        return self._contains_tumor

    def contains_nontumor_without_nerve(self, threshold: float = 0.5) -> bool:
        if self._contains_nontumor_without_nerve is not None:
            return self._contains_nontumor_without_nerve
        empty_pixels = self.count_mask_pixels(self.EMPTY_MASK_COLOR)
        self._contains_nontumor_without_nerve = (
            empty_pixels / (self.width * self.height) > threshold
        )
        return self._contains_nontumor_without_nerve

    def contains_pni(self, threshold_pixels: int = 100) -> bool:
        if self._contains_pni is not None:
            return self._contains_pni
        pni_pixels = self.count_mask_pixels(self.PNI_MASK_COLOR)
        self._contains_pni = pni_pixels > threshold_pixels
        return self._contains_pni

    def count_mask_pixels(self, color=Tuple[int, int, int]) -> int:
        dst = cv.inRange(self.read_block(), color, color)
        return cv.countNonZero(dst)


class Sample:
    def __init__(self, image: Image, mask: Mask = None):
        self._image = image
        if mask is not None:
            self._mask = mask
            # Mask has lower resolution than image, so resizing will be required while manipulating with it
            self._mask_image_width_ratio = self._mask.width / self._image.width
            self._mask_image_height_ratio = self._mask.height / self._image.height

    @property
    def image(self):
        return self._image

    @property
    def mask(self):
        return self._mask

    @property
    def mask_image_width_ratio(self):
        return self._mask_image_width_ratio

    @property
    def mask_image_height_ratio(self):
        return self._mask_image_height_ratio

    def is_purple(
        self,
        rect: Tuple[int, int, int, int] = (0, 0, 0, 0),
        purple_threshold: int = 220,
        purple_ratio: float = 0.5,
    ) -> bool:
        patch = self.image.read_block(rect=rect)
        gray_patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        _, black_white_patch = cv.threshold(
            gray_patch, purple_threshold, 255, cv.THRESH_BINARY_INV
        )
        pixels = black_white_patch.size
        colored_pixels = cv.countNonZero(black_white_patch)
        return colored_pixels / pixels >= purple_ratio

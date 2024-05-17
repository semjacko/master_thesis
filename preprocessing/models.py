from typing import Tuple, Union

import cv2 as cv
import numpy as np
import slideio


class Image:
    def __init__(self, image: Union[np.ndarray, str]):
        if type(image) is np.ndarray:
            self._raw_image = image
        elif type(image) is str:
            self._raw_image = cv.imread(image)
        self._width = self._raw_image.shape[1]
        self._height = self._raw_image.shape[0]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def read_block(
        self,
        rect: Tuple[int, int, int, int] = None,  # x, y, w, h
        size: Tuple[int, int] = None,  # w, h
    ) -> np.ndarray:
        block = self._raw_image.copy()
        if rect is not None:
            block = self._raw_image[
                rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]
            ]
        if size is not None:
            return cv.resize(block, size, 0, 0, interpolation=cv.INTER_NEAREST)
        return block


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

    def contains_nerve(self, threshold: float = 0.05) -> bool:
        if self._contains_nerve is not None:
            return self._contains_nerve
        nerve_pixels = self.count_mask_pixels(self.NERVE_MASK_COLOR)
        self._contains_nerve = nerve_pixels / (self.width * self.height) > threshold
        return self._contains_nerve

    def contains_tumor(self, threshold: float = 0.8) -> bool:
        if self._contains_tumor is not None:
            return self._contains_tumor
        tumor_pixels = self.count_mask_pixels(self.TUMOR_MASK_COLOR)
        self._contains_tumor = tumor_pixels / (self.width * self.height) > threshold
        return self._contains_tumor

    def contains_nontumor_without_nerve(self, threshold: float = 0.8) -> bool:
        if self._contains_nontumor_without_nerve is not None:
            return self._contains_nontumor_without_nerve
        empty_pixels = self.count_mask_pixels(self.EMPTY_MASK_COLOR)
        self._contains_nontumor_without_nerve = (
            empty_pixels / (self.width * self.height) > threshold
        )
        return self._contains_nontumor_without_nerve

    def contains_pni(self, threshold_pixels: int = 1500) -> bool:
        if self._contains_pni is not None:
            return self._contains_pni
        pni_pixels = self.count_mask_pixels(self.PNI_MASK_COLOR)
        self._contains_pni = pni_pixels > threshold_pixels
        return self._contains_pni

    def count_mask_pixels(self, color=Tuple[int, int, int]) -> int:
        dst = cv.inRange(self.read_block(), color, color)
        return cv.countNonZero(dst)


class SVS(Image):
    def __init__(self, scene: Union[slideio.Scene, str]):
        if type(scene) is slideio.Scene:
            self._scene = scene
        elif type(scene) is str:
            slide = slideio.open_slide(path=scene, driver="SVS")
            self._scene = slide.get_scene(0)
        self._width = self._scene.rect[2]
        self._height = self._scene.rect[3]

    def read_block(
        self,
        rect: Tuple[int, int, int, int] = (0, 0, 0, 0),
        size: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        return self._scene.read_block(rect=rect, size=size)
    
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

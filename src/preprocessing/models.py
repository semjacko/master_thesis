import slideio
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from typing import Iterator, Tuple, Union


class Image:
    def __init__(self, image: Union[str, np.ndarray]):
        if type(image) is str:
            self._image = cv.imread(image)
        elif type(image) is np.ndarray:
            self._image = image
        self._width = self._image.shape[1]
        self._height = self._image.shape[0]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def read_block(
        self,
        rect: Tuple[int, int, int, int] = (0, 0, 0, 0),
        size: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        block = self._image.copy()
        if rect != (0, 0, 0, 0):
            block = self._image[
                rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]
            ]
        if size != (0, 0):
            return cv.resize(block, size, 0, 0, interpolation=cv.INTER_NEAREST)
        return block


class Mask(Image):
    NERVE_MASK_COLOR = (1, 1, 1)
    NERVE_BOX_MASK_COLOR = (11, 11, 11)
    PERI_INV_MASK_COLOR = (2, 2, 2)
    TUMOR_MASK_COLOR = (13, 13, 13)
    EMPTY_MASK_COLOR = (14, 14, 14)

    MASK_COLOR_MAPPING = {
        NERVE_MASK_COLOR: (255, 88, 0),  # ORANGE nerve without tumor
        NERVE_BOX_MASK_COLOR: (255, 255, 0),  # YELLOW nerve without tumor box
        PERI_INV_MASK_COLOR: (255, 0, 0),  # RED perineural invasion
        TUMOR_MASK_COLOR: (0, 255, 0),  # GREEN tumor without nerve box
        EMPTY_MASK_COLOR: (0, 0, 255),  # BLUE non-tumor without nerve box
    }

    def thumb(self, max_width: int = 512) -> np.ndarray:
        w = max_width if self.width > max_width else self.width
        h = int(self.height * w / self.width)
        image = self.read_block(size=(w, h))
        for mask_color, res_color in self.MASK_COLOR_MAPPING.items():
            image[np.all(image == mask_color, axis=-1)] = res_color
        return image

    # Returns an image overlaid by mask contours
    def overlay(self, image_to_overlay: np.ndarray) -> np.ndarray:
        w, h = image_to_overlay.shape[1], image_to_overlay.shape[0]
        mask = cv.resize(self._image, (w, h), 0, 0, interpolation=cv.INTER_NEAREST)
        for mask_color, contour_color in self.MASK_COLOR_MAPPING.items():
            if mask_color == self.NERVE_MASK_COLOR:
                # Contours for nerves will be obtained from tumor box
                continue
            m = cv.inRange(mask, mask_color, mask_color)
            contours, _ = cv.findContours(m, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image_to_overlay, contours, -1, contour_color, 2)
        return image_to_overlay

    def contains_nerve(self, threshold: float = 0.1) -> bool:
        nerve_pixels = self.number_nerve_pixels()
        return nerve_pixels / (self._width * self.height) > threshold

    def number_nerve_pixels(self) -> int:
        dst = cv.inRange(self._image, self.NERVE_MASK_COLOR, self.NERVE_MASK_COLOR)
        return cv.countNonZero(dst)

    def contains_tumor(self) -> bool:
        pass

    def contains_peri_inv(self) -> bool:
        pass


class SVS(Image):
    def __init__(self, scene: Union[str, slideio.Scene]):
        if type(scene) is str:
            slide = slideio.open_slide(path=scene, driver="SVS")
            self._scene = slide.get_scene(0)
        elif type(scene) is slideio.Scene:
            self._scene = scene
        self._width = self._scene.rect[2]
        self._height = self._scene.rect[3]

    def read_block(
        self,
        rect: Tuple[int, int, int, int] = (0, 0, 0, 0),
        size: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        return self._scene.read_block(rect=rect, size=size)


class Sample:
    def __init__(self, slide: Union[str, slideio.Scene, np.ndarray], mask: Mask = None):
        # Load whole slide scene
        if type(slide) is np.ndarray:
            self._image = Image(image=slide)
        elif type(slide) in [str, slideio.Scene]:
            self._image = SVS(scene=slide)
        # Load mask
        if mask is not None:
            self._mask = mask
            # Mask has lower resolution than WSI, so resizing will be
            #  required while manipulating with it
            self._mask_wsi_width_ratio = self._mask.width / self._image.width
            self._mask_wsi_height_ratio = self._mask.height / self._image.height

    @property
    def image(self):
        return self._image

    @property
    def mask(self):
        return self._mask

    @property
    def mask_wsi_width_ratio(self):
        return self._mask_wsi_width_ratio

    @property
    def mask_wsi_height_ratio(self):
        return self._mask_wsi_height_ratio

    # Returns thumbnail of the WSI
    def thumb(self, mask_overlay: bool = False, max_width: int = 1024) -> np.ndarray:
        w = max_width if self.image.width > max_width else self.image.width
        h = int(self.image.height * w / self.image.width)
        image = self.image.read_block(size=(w, h))
        if mask_overlay and self.mask:
            return self.mask.overlay(image)
        return image

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


class PatchGenerator:
    def __init__(self, wsi: Sample):
        self._wsi = wsi

    def generate_patches(
        self, patch_size: Tuple[int, int], overlap: float = 0.5
    ) -> Iterator[Sample]:
        step_size = int(patch_size[0] * (1 - overlap))
        w, h = patch_size[0], patch_size[1]
        mask_w = int(w * self._wsi.mask_wsi_width_ratio)
        mask_h = int(h * self._wsi.mask_wsi_height_ratio)
        for y in range(0, self._wsi.image.height - h + 1, step_size):
            print(f"{y}/{self._wsi.image.height}")

            mask_y = int(y * self._wsi.mask_wsi_height_ratio)
            if not self.check_row_prerequisities(mask_y, mask_h):
                continue

            for x in range(0, self._wsi.image.width - w + 1, step_size):
                mask_x = int(x * self._wsi.mask_wsi_width_ratio)
                mask_block = self._wsi.mask.read_block(
                    rect=(mask_x, mask_y, mask_w, mask_h), size=(w, h)
                )
                mask = Mask(mask_block)
                if not mask.contains_nerve(threshold=0.07):
                    continue
                image = self._wsi.image.read_block(rect=(x, y, w, h))
                patch = Sample(image, mask)
                yield patch

    def check_row_prerequisities(self, mask_y: int, mask_h: int) -> bool:
        mask_block = self._wsi.mask.read_block(
            rect=(0, mask_y, self._wsi._mask._width, mask_h)
        )
        mask = Mask(mask_block)
        return mask.number_nerve_pixels() > 0

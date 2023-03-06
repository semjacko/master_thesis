import slideio
import numpy as np
import cv2 as cv
from typing import Tuple, Union


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

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


class WandbConfig:
    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        image_size: Tuple[int, int] = (512, 512),
        architecture: str = "resnet50",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.image_size = image_size
        self.architecture = architecture


class Config(WandbConfig):
    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        dataset_size: int = 0,
        image_size: Tuple[int, int] = (512, 512),
        architecture: str = "resnet50",
        buffer_size: int = 512,
    ) -> None:
        super().__init__(epochs, batch_size, split_ratio, image_size, architecture)
        self.dataset_size = dataset_size
        self.train_size = int(self.ds_size * self.split_ratio[0])
        self.val_size = int(self.ds_size * self.split_ratio[1])
        self.test_size = int(self.ds_size * self.split_ratio[2])
        self.steps_per_epoch = self.train_size // batch_size
        self.val_steps = self.val_size // batch_size
        self.buffer_size = buffer_size
        self.data_path_colon = "/kaggle/input/nerves/colon/"
        self.data_path_pancreas = "/kaggle/input/nerves/pancreas/"
        self.data_path_prostate = "/kaggle/input/nerves/prostate/"

from typing import Tuple


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

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List


# Loads all paths to images and their corresponding masks and returns them as 2 tensors
def load_image_and_mask_paths(data_path: str) -> Tuple[List[str], List[str]]:
    image_folder_path = data_path + "images/"
    mask_folder_path = data_path + "masks/"

    ids = [os.path.splitext(image)[0] for image in os.listdir(image_folder_path)]

    image_paths = [image_folder_path + id + ".jpg" for id in ids]
    mask_paths = [mask_folder_path + id + ".png" for id in ids]

    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)

    return image_paths, mask_paths


# Returns the function that loads a single image and its corresponding
# mask from the provided paths
def load_images_and_masks(
    image_size: Tuple[int, int] = (512, 512)
) -> Callable[[str, str], Tuple[tf.Tensor, tf.Tensor]]:
    def load_images_func(
        image_path: str, mask_path: str
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)

        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, channels=1) // 255
        mask = tf.image.resize(
            mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return image, mask

    return load_images_func


# Splits the dataset into 2 parts based on the train_ratio and returns these 2 parts
def split(
    dataset: tf.data.Dataset, train_ratio: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_len = dataset.cardinality().numpy()
    train_size = int(ds_len * train_ratio)
    shuffle_ds = dataset.shuffle(buffer_size=500)  # TODO
    return shuffle_ds.take(train_size), shuffle_ds.skip(train_size)


# Displays the image, mask and possibly the prediction if provided in a single plot
def show(image: np.array, mask: np.array, prediction: np.array = None) -> None:
    display_list = [image, mask] if prediction is None else [image, mask, prediction]
    titles = ["Input Image", "True Mask", "Predicted Mask"]
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    plt.show()

import os
from IPython.display import clear_output
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Callable


# Loads all paths to images and their corresponding masks and returns them as 2 tensors
def load_image_and_mask_paths(data_path: str) -> Tuple[list[str], list[str]]:
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


# TODO
# class Augment(tf.keras.layers.Layer):
#     def __init__(self, seed=42):
#         super().__init__()
#         # both use the same seed, so they'll make the same random changes.
#         self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
#         self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

#     def call(self, inputs, labels):
#         inputs = self.augment_inputs(inputs)
#         labels = self.augment_labels(labels)
#         return inputs, labels


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


# TODO move this to model
def show_prediction(
    model, image: np.array, mask: np.array, binary: bool = False
) -> None:  # TODO input params
    prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    show(image, mask, tf.math.round(prediction)) if binary else show(
        image, mask, prediction
    )


# TODO
# https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
def balanced_cross_entropy(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (
            weight_a + weight_b
        ) + y_pred * weight_b
        return tf.reduce_mean(o)

    return loss


# TODO
def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred**gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
            weight_a + weight_b
        ) + logits * weight_b

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(
            logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred
        )

        return tf.reduce_mean(loss)

    return loss


# TODO
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


# TODO
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)  # TODO
        show_prediction()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))

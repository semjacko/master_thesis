import tensorflow as tf
import keras.backend as K

# TODO
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


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


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
def dice_loss(targets, inputs, smooth = 1e-6):
    targets = tf.cast(targets, tf.float32)
    intersection = K.sum(targets * inputs)
    
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)

    return 1 - dice


def dice_bce_loss(targets, inputs, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = tf.keras.losses.BinaryCrossentropy()(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (
        K.sum(targets) + K.sum(inputs) + smooth
    )
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def iou_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

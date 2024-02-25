import cv2 as cv
import numpy as np
import tensorflow as tf

from preprocessing.models import Image
from preprocessing.preprocessor import Preprocessor


class _PredictionMask:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._total = np.zeros(shape=(self.height, self.width))
        self._count = np.zeros(shape=(self.height, self.width))

    def add_part(self, mask: Image, x: int, y: int):
        assert x + mask.width < self._width and y + mask.height < self._height
        self._total[y:y + mask.height, x:x + mask.width] += mask.read_block()
        self._count[y:y + mask.height, x:x + mask.width] += 1

    @property
    def result_mask(self) -> Image:
        return Image(self._total / self._count)


class Prediction:
    def __init__(self, image: Image):
        self._image = image
        self._nerve_mask = _PredictionMask(image.width, image.height)
        self._pni_mask = _PredictionMask(image.width, image.height)

    def add_nerves(self, mask: Image, x: int, y: int):
        self._nerve_mask.add_part(mask, x, y)

    def add_pni(self, mask: Image, x: int, y: int):
        self._pni_mask.add_part(mask, x, y)

    def show(self):
        # TODO: Overlay image with nerve 
        nerves = self._nerve_mask.result_mask
        pni = self._pni_mask.result_mask
    

class Processor:
    _overlap = 0.5
    _patch_size=(1024, 1024)

    def __init__(self, tumor_model_path: str, nerve_model_path: str, pni_model_path: str) -> None:
        self._preprocessor = Preprocessor(patch_size=self._patch_size, overlap=self._overlap) # TODO: Create new Preprocessor that extracts all patches
        self._tumor_detector: tf.keras.Model = tf.keras.models.load_model(tumor_model_path)
        self._nerve_segmenter: tf.keras.Model = tf.keras.models.load_model(nerve_model_path)
        self._pni_segmenter: tf.keras.Model = tf.keras.models.load_model(pni_model_path)

    def process(self, image: Image) -> Prediction:
        prediction = Prediction(image=image)
        
        # Extract patches
        for (x, y, sample) in self._preprocessor.extract_patches(image):
            is_tumor = self.detect_tumor(sample.image)
            nerve_mask = self.segment_nerves(sample.image)            
            if is_tumor and self._contains_nerves(nerve_mask):
                pni_mask = self.segment_pni(sample.image, nerve_mask)
                prediction.add_pni(pni_mask, x=x, y=y)
            prediction.add_nerves(nerve_mask, x=x, y=y)

        return prediction

    def segment_nerves(self, image: Image) -> Image:
        input_image = tf.expand_dims(image.read_block(), axis=0)
        prediction: np.ndarray = self._nerve_segmenter.predict(input_image)
        return Image(prediction[0])

    def _contains_nerves(nerve_mask: Image, threshold=0.05) -> bool:
        nerve_pixels = cv.countNonZero(nerve_mask.read_block().round())
        return nerve_pixels / (nerve_mask.width * nerve_mask.height) > threshold

    def detect_tumor(self, image: Image) -> bool:
        input_image = tf.expand_dims(image.read_block(), axis=0)
        prediction: float = self._tumor_detector.predict(input_image)
        return round(prediction[0]) > 0
     
    def segment_pni(self, image: Image, nerve_mask: Image) -> Image:
        input_image = tf.concat([image.read_block(), nerve_mask.read_block()], axis=-1)
        input_image = tf.expand_dims(input_image, axis=0)
        prediction: np.ndarray = self._pni_segmenter.predict(input_image)
        return Image(prediction[0])

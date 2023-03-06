import os
import cv2 as cv

from processor.preprocessing.preprocessor import Preprocessor
from processor.tumor_classification.tumor_classifier import TumorClassifier
from processor.nerve_segmentation.nerve_segmenter import NerveSegmenter
from processor.pni_segmentation.pni_segmenter import PNISegmenter
from processor.postprocessing.postprocessor import Postprocessor
from common.models import Image, SVS


class Processor:
    def __init__(self, tumor_model_path, nerve_model_path, pni_model_path) -> None:
        self.preprocessor = Preprocessor(patch_size=(1024, 1024), overlap=0.5)
        self.tumor_classifier = TumorClassifier(tumor_model_path)
        self.nerve_segmenter = NerveSegmenter(nerve_model_path)
        self.pni_segmenter = PNISegmenter(pni_model_path)
        self.postprocessor = Postprocessor()

    def process(self, svs: SVS):
        # Extract patches
        patches_path = self.preprocessor.extract_patches(svs)

        # Classify tumors
        tumor_classification = {}
        for path in os.listdir(patches_path):
            image = Image(path)
            tumor_classification[path] = self.tumor_classifier.classify(image)
        tumor_output_path = "./tumor_classifications.txt"
        with open(tumor_output_path, "w") as f:
            print(tumor_classification, file=f)

        # Segment nerves
        nerve_output_path = "./nerves_masks"
        for path in os.listdir(patches_path):
            image = Image(path)
            nerves_mask = self.nerve_segmenter.segment(image)
            cv.imwrite(nerve_output_path + path, nerves_mask)

        # Segment pni
        pni_output_path = "./pni_masks"
        for path in os.listdir(patches_path):
            image = Image(path)
            nerves_mask = Image(nerve_output_path + path)
            pni_mask = self.pni_segmenter.segment(image, nerves_mask)
            cv.imwrite(pni_output_path + path, pni_mask)

        # Postprocess
        output = self.postprocessor.process(
            tumor_output_path, nerve_output_path, pni_output_path
        )

        return output

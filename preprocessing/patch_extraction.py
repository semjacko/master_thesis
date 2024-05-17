import glob
import os

import cv2 as cv
import numpy as np

from models import Mask, SVS
from preprocessor import NervePreprocessor, PNIPreprocessor, TumorPreprocessor


def extract_nerves(sample_name: str, svs: SVS, mask: Mask):
    preprocessor = NervePreprocessor(patch_size=(1024, 1024), overlap=0.5)
    data_path = "../data/pancreas/nerves/"
    os.makedirs(data_path + "images/", exist_ok=True)
    os.makedirs(data_path + "masks/", exist_ok=True)

    for i, (x, y, sample) in enumerate(preprocessor.extract_patches(svs, mask)):
        print(f"\t{i+1}. witdh:{x}/{svs.width} height:{y}/{svs.height}")
        cv.imwrite(
            f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}.jpg",
            sample.image.read_block(),
        )
        cv.imwrite(
            f"{data_path}masks/{sample_name}_{str(i+1).zfill(4)}.png",
            cv.inRange(
                sample.mask.read_block(),
                Mask.NERVE_MASK_COLOR,
                Mask.NERVE_MASK_COLOR,
            ),
        )


def extract_tumors(sample_name: str, svs: SVS, mask: Mask):
    preprocessor = TumorPreprocessor(patch_size=(1024, 1024), overlap=0)
    data_path = "../data/pancreas/tumors/"
    os.makedirs(data_path + "images/", exist_ok=True)

    for i, (x, y, sample) in enumerate(preprocessor.extract_patches(svs, mask)):
        print(f"\t{i+1}. witdh:{x}/{svs.width} height:{y}/{svs.height}")
        label = "tumor" if sample.mask.contains_tumor() else "empty"
        cv.imwrite(
            f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}_{label}.jpg",
            sample.image.read_block(),
        )


def extract_pni(sample_name: str, svs: SVS, mask: Mask):
    preprocessor = PNIPreprocessor(patch_size=(1024, 1024), overlap=0.5)
    data_path = "../data/pancreas/pni/"
    os.makedirs(data_path + "images/", exist_ok=True)
    os.makedirs(data_path + "masks/", exist_ok=True)

    for i, (x, y, sample) in enumerate(preprocessor.extract_patches(svs, mask)):
        print(f"\t{i+1}. witdh:{x}/{svs.width} height:{y}/{svs.height}")
        cv.imwrite(
            f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}.jpg",
            sample.image.read_block(),
        )
        cv.imwrite(
            f"{data_path}masks/{sample_name}_{str(i+1).zfill(4)}.png",
            cv.inRange(
                sample.mask.read_block(),
                Mask.PNI_MASK_COLOR,
                Mask.PNI_MASK_COLOR,
            ),
        )


if __name__ == "__main__":
    data_raw_path = "../data_raw/pancreas/"
    for svs_file in glob.glob(data_raw_path + "*.svs"):
        sample_name = os.path.splitext(os.path.basename(svs_file))[0]
        
        svs = SVS(f"{data_raw_path}{sample_name}.svs")
        mask = Mask(f"{data_raw_path}{sample_name}_l1_mask.tif")

        print(f"Processing: {sample_name}, extracting nerves...")
        extract_nerves(sample_name, svs, mask)
        print(f"Processing: {sample_name}, extracting tumors...")
        extract_tumors(sample_name, svs, mask)
        print(f"Processing: {sample_name}, extracting PNI...")
        extract_pni(sample_name, svs, mask)

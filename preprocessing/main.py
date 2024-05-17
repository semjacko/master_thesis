import glob
import os

import cv2 as cv
import numpy as np

from preprocessing.models import Mask, SVS
from preprocessing.preprocessor import (
    NervePreprocessor,
    PNIPreprocessor,
    TumorPreprocessor
)


def extract_nerves(sample_name: str, svs: SVS, mask: Mask):
    preprocessor = NervePreprocessor(patch_size=(1024, 1024), overlap=0.5)
    data_path = "./data/pancreas/nerves/"

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
    data_path = "./data/pancreas/tumors/"

    for i, (x, y, sample) in enumerate(preprocessor.extract_patches(svs, mask)):
        print(f"\t{i+1}. witdh:{x}/{svs.width} height:{y}/{svs.height}")
        label = "tumor" if sample.mask.contains_tumor() else "empty"
        cv.imwrite(
            f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}_{label}.jpg",
            sample.image.read_block(),
        )


def extract_pni(sample_name: str, svs: SVS, mask: Mask):
    preprocessor = PNIPreprocessor(patch_size=(1024, 1024), overlap=0.5)
    data_path = "./data/pancreas/pni/"

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


def main():
    data_raw_path = "./data_raw/pancreas/"
    for svs_file in glob.glob(data_raw_path + "*.svs"):
        sample_name = os.path.splitext(os.path.basename(svs_file))[0]
        print(f"Processing: {sample_name}")
        
        svs = SVS(f"{data_raw_path}{sample_name}.svs")
        mask = Mask(f"{data_raw_path}{sample_name}_l1_mask.tif")

        extract_nerves(sample_name, svs, mask)



if __name__ == "__main__":
    main()

    # svs = SVS("./data_raw/pancreas/Pan_PNI2021chall_train_0043.svs")
    # mask = Mask("./data_raw/pancreas/Pan_PNI2021chall_train_0043_l1_mask.tif")
    
    # x = 33792
    # y = 16384
    # height = 10240
    # width = 10240

    # image_patch = svs.read_block(rect=(x, y, height, width))
    # mask_patch = mask.read_block(rect=(x//4, y//4, height//4, width//4), size=(height, width))

    # cv.imwrite("test_image.jpg", image_patch)
    # cv.imwrite("test_mask.png", mask_patch)

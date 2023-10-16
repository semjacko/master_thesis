from common.models import SVS
from preprocessing.preprocessor import NervePreprocessor
from preprocessing.models import Mask
import cv2 as cv
import glob
import os


def main():
    extract_nerves()


def extract_nerves():
    preprocessor = NervePreprocessor(patch_size=(1024, 1024), overlap=0.5)

    data_raw_path = "./data_raw/pancreas/"
    data_path = "./data/pancreas/nerves/"
    for svs_file in glob.glob(data_raw_path + "*.svs"):
        sample_name = os.path.splitext(os.path.basename(svs_file))[0]
        print(f"Processing: {sample_name}")
        svs = SVS(f"{data_raw_path}{sample_name}.svs")
        mask = Mask(f"{data_raw_path}{sample_name}_l1_mask.tif")
        for i, sample_with_label in enumerate(preprocessor.extract_patches(svs, mask)):
            print(i)
            print(sample_with_label[0])
            print(sample_with_label[1])
            # cv.imwrite(
            #     f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}_empty.jpg",
            #     sample.image.read_block(),
            # )
            # cv.imwrite(
            #     f"{data_path}masks/{sample_name}_{str(i+1).zfill(4)}.png",
            #     cv.inRange(
            #         sample.mask.read_block(),
            #         Mask.TUMOR_MASK_COLOR,
            #         Mask.TUMOR_MASK_COLOR,
            #     ),
            # )


def extract_tumors():
    pass


def extract_pni():
    pass


if __name__ == "__main__":
    main()

from models import Mask, Image, Sample, PatchGenerator
import cv2 as cv
import glob
import os


if __name__ == "__main__":
    # cv.imshow("td", mask.overlay(image._image))
    # cv.waitKey(0)

    data_raw_path = "./data_raw/pancreas/"
    data_path = "./data/pancreas/"

    for svs_file in glob.glob(data_raw_path + "*.svs"):
        sample_name = os.path.splitext(os.path.basename(svs_file))[0]
        print(f"Processing: {sample_name}")
        wsi = Sample(
            slide=f"{data_raw_path}{sample_name}.svs",
            mask=Mask(f"{data_raw_path}{sample_name}_l1_mask.tif"),
        )
        patch_gen = PatchGenerator(wsi)
        for i, patch in enumerate(patch_gen.generate_patches(patch_size=(1024, 1024))):
            cv.imwrite(
                f"{data_path}images/{sample_name}_{str(i+1).zfill(4)}.jpg",
                patch.image.read_block(),
            )
            cv.imwrite(
                f"{data_path}masks/{sample_name}_{str(i+1).zfill(4)}.png",
                cv.inRange(
                    patch.mask.read_block(),
                    Mask.NERVE_MASK_COLOR,
                    Mask.NERVE_MASK_COLOR,
                ),
            )

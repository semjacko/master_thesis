from typing import Tuple, Iterator, Union

from common.models import Image, SVS
from preprocessing.models import Mask, Sample


class Preprocessor:
    def __init__(self, patch_size: Tuple[int, int], overlap: float) -> None:
        self._patch_size = patch_size
        self._overlap = overlap
        self._step_size = int(patch_size[0] * (1 - overlap))

    def extract_patches(
        self, svs: SVS, mask: Mask = None
    ) -> Iterator[Tuple[Sample, str]]:
        sample = Sample(svs, mask)
        return self._generate_patches(sample)

    def _generate_patches(self, sample: Sample) -> Iterator[Tuple[Sample, str]]:
        w, h = self._patch_size[0], self._patch_size[1]
        mask_w = int(w * sample.mask_image_width_ratio)
        mask_h = int(h * sample.mask_image_height_ratio)
        for y in range(0, sample.image.height - h + 1, self._step_size):
            print(f"{y}/{sample.image.height}")

            mask_y = int(y * sample.mask_image_height_ratio)
            if not self._check_row_prerequisities(sample.mask, mask_y, mask_h):
                continue

            for x in range(0, sample.image.width - w + 1, self._step_size):
                mask_x = int(x * sample.mask_image_width_ratio)
                mask_block = sample.mask.read_block(
                    rect=(mask_x, mask_y, mask_w, mask_h), size=(w, h)
                )
                mask = Mask(mask_block)

                label = self._label_mask(mask)
                if not label:
                    continue

                image_block = sample.image.read_block(rect=(x, y, w, h))
                patch = Sample(Image(image_block), mask)
                yield patch, label

    def _check_row_prerequisities(self, mask: Mask, mask_y: int, mask_h: int) -> bool:
        row_mask_block = mask.read_block(rect=(0, mask_y, mask.width, mask_h))
        row_mask = Mask(row_mask_block)
        return self._check_row(row_mask)

    def _check_row(self, mask: Mask) -> bool:
        raise NotImplementedError

    def _label_mask(self, mask: Mask) -> Union[str, None]:
        raise NotImplementedError


class NervePreprocessor(Preprocessor):
    def _check_row(self, mask: Mask):
        return mask.count_mask_pixels(mask.NERVE_MASK_COLOR) > 0

    def _label_mask(self, mask: Mask) -> Union[str, None]:
        if mask.contains_nerve():
            return "nerve"
        return None


class TumorPreprocessor(Preprocessor):
    def _check_row(self, mask: Mask):
        # Check if there are some tumor, nerve (without tumor) or empty pixels
        return (
            mask.count_mask_pixels(mask.TUMOR_MASK_COLOR) > 0
            or mask.count_mask_pixels(mask.NERVE_MASK_COLOR) > 0
            or mask.count_mask_pixels(mask.EMPTY_MASK_COLOR) > 0
        )

    def _label_mask(self, mask: Mask) -> Union[str, None]:
        if mask.contains_tumor():
            return "tumor"
        elif mask.contains_nerve():
            return "nerve"
        elif mask.contains_nontumor_without_nerve():
            return "empty"
        return None


class PNIPreprocessor(Preprocessor):
    def _check_row(self, mask: Mask):
        return mask.count_mask_pixels(mask.PNI_MASK_COLOR) > 0

    def _label_mask(self, mask: Mask) -> Union[str, None]:
        if mask.contains_pni():
            return "pni"
        return None

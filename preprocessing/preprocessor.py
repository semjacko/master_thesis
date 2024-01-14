from typing import Iterator, Tuple

from preprocessing.models import Image, Mask, Sample


class Preprocessor:
    def __init__(self, patch_size: Tuple[int, int], overlap: float) -> None:
        self._patch_size = patch_size
        self._overlap = overlap
        self._step_size = int(patch_size[0] * (1 - overlap))

    # TODO: Return also patch #number
    def extract_patches(self, image: Image, mask: Mask = None) -> Iterator[Sample]:
        sample = Sample(image, mask)
        return self._generate_patches(sample)

    def _generate_patches(self, sample: Sample) -> Iterator[Sample]:
        w, h = self._patch_size[0], self._patch_size[1]
        mask_w = int(w * sample.mask_image_width_ratio)
        mask_h = int(h * sample.mask_image_height_ratio)
        for y in range(0, sample.image.height - h + 1, self._step_size):
            print(f"{y}/{sample.image.height}")

            mask_y = int(y * sample.mask_image_height_ratio)
            if not self._check_row_prerequisites(sample.mask, mask_y, mask_h):
                continue

            for x in range(0, sample.image.width - w + 1, self._step_size):
                mask_x = int(x * sample.mask_image_width_ratio)
                mask_block = sample.mask.read_block(
                    rect=(mask_x, mask_y, mask_w, mask_h), size=(w, h)
                )
                mask = Mask(mask_block)
                if not self._check_patch_prerequisites(mask):
                    continue
                image_block = sample.image.read_block(rect=(x, y, w, h))
                patch = Sample(Image(image_block), mask)
                yield patch

    def _check_row_prerequisites(self, mask: Mask, mask_y: int, mask_h: int) -> bool:
        row_mask_block = mask.read_block(rect=(0, mask_y, mask.width, mask_h))
        row_mask = Mask(row_mask_block)
        return self._check_row(row_mask)

    def _check_row(self, mask: Mask) -> bool:
        raise NotImplementedError

    def _check_patch_prerequisites(self, mask: Mask) -> bool:
        raise NotImplementedError


class NervePreprocessor(Preprocessor):
    def _check_row(self, mask: Mask) -> bool:
        return mask.count_mask_pixels(mask.NERVE_MASK_COLOR) > 0

    def _check_patch_prerequisites(self, mask: Mask) -> bool:
        return mask.contains_nerve()


class TumorPreprocessor(Preprocessor):
    def __init__(self, patch_size: Tuple[int, int], overlap: float) -> None:
        super().__init__(patch_size, overlap)
        self._tumors = 0
        self._nerves = 0
        self._empty = 0

    def _check_row(self, mask: Mask) -> bool:
        # Check if there are some tumors
        if mask.count_mask_pixels(mask.TUMOR_MASK_COLOR) > 0:
            return True
        # Check if number of extracted nerves is less than half of the tumors and if there are some nerves
        if (
            self._tumors // 2 > self._nerves
            and mask.count_mask_pixels(mask.NERVE_MASK_COLOR) > 0
        ):
            return True
        # Check if number of extracted empty is less than half of the tumors and if there are some empty
        if (
            self._tumors // 2 > self._empty
            and mask.count_mask_pixels(mask.EMPTY_MASK_COLOR) > 0
        ):
            return True
        return False

    def _check_patch_prerequisites(self, mask: Mask) -> bool:
        if mask.contains_tumor():
            self._tumors += 1
            return True
        if mask.contains_nerve():
            self._nerves += 1
            return True
        if mask.contains_nontumor_without_nerve():
            self._empty += 1
            return True
        return False


class PNIPreprocessor(Preprocessor):
    def _check_row(self, mask: Mask) -> bool:
        return mask.count_mask_pixels(mask.PNI_MASK_COLOR) > 0

    def _check_patch_prerequisites(self, mask: Mask) -> bool:
        return mask.contains_pni()

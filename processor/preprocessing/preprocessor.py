from typing import Tuple, Iterator

from common.models import SVS
from processor.preprocessing.models import Mask, Sample


class Preprocessor:
    def __init__(self, patch_size: Tuple[int, int], overlap: float) -> None:
        self._patch_size = patch_size
        self._overlap = overlap
        self._step_size = int(patch_size[0] * (1 - overlap))

    def extract_patches(self, svs: SVS, mask: Mask = None):
        sample = Sample(svs, mask)
        return self._generate_patches(sample)

    # TODO: Generate patches also without mask

    def _generate_patches(self, sample: Sample) -> Iterator[Sample]:

        w, h = self._patch_size[0], self._patch_size[1]
        mask_w = int(w * sample.mask_image_width_ratio)
        mask_h = int(h * sample.mask_image_height_ratio)
        for y in range(0, sample.image.height - h + 1, self._step_size):
            print(f"{y}/{sample.image.height}")

            mask_y = int(y * sample.mask_image_height_ratio)
            if not self.check_row_prerequisities(mask_y, mask_h):
                continue

            for x in range(0, sample.image.width - w + 1, self._step_size):
                mask_x = int(x * sample.mask_image_width_ratio)
                mask_block = sample.mask.read_block(
                    rect=(mask_x, mask_y, mask_w, mask_h), size=(w, h)
                )
                mask = Mask(mask_block)
                if not mask.contains_nerve(threshold=0.07):
                    continue
                image = sample.image.read_block(rect=(x, y, w, h))
                patch = Sample(image, mask)
                yield patch

    # TODO
    def check_row_prerequisities(self, mask: Mask, mask_y: int, mask_h: int) -> bool:
        mask_block = mask.read_block(rect=(0, mask_y, mask.width, mask_h))
        mask = Mask(mask_block)
        return mask.count_mask_pixels(mask.NERVE_MASK_COLOR) > 0

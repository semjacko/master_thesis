# class Mask(Image):
#     MASK_COLOR_MAPPING = {
#         NERVE_MASK_COLOR: (255, 88, 0),  # ORANGE nerve without tumor
#         NERVE_BOX_MASK_COLOR: (255, 255, 0),  # YELLOW nerve without tumor box
#         PNI_MASK_COLOR: (255, 0, 0),  # RED perineural invasion
#         TUMOR_MASK_COLOR: (0, 255, 0),  # GREEN tumor without nerve box
#         EMPTY_MASK_COLOR: (0, 0, 255),  # BLUE non-tumor without nerve box
#     }

#     def thumb(self, max_width: int = 512) -> np.ndarray:
#         w = max_width if self.width > max_width else self.width
#         h = int(self.height * w / self.width)
#         image = self.read_block(size=(w, h))
#         for mask_color, res_color in self.MASK_COLOR_MAPPING.items():
#             image[np.all(image == mask_color, axis=-1)] = res_color
#         return image

#     # Returns an image overlaid by mask contours
#     def overlay(self, image_to_overlay: np.ndarray) -> np.ndarray:
#         w, h = image_to_overlay.shape[1], image_to_overlay.shape[0]
#         mask = cv.resize(self._image, (w, h), 0, 0, interpolation=cv.INTER_NEAREST)
#         for mask_color, contour_color in self.MASK_COLOR_MAPPING.items():
#             if mask_color == self.NERVE_MASK_COLOR:
#                 # Contours for nerves will be obtained from tumor box
#                 continue
#             m = cv.inRange(mask, mask_color, mask_color)
#             contours, _ = cv.findContours(m, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#             cv.drawContours(image_to_overlay, contours, -1, contour_color, 2)
#         return image_to_overlay


# class Sample:
#     Returns thumbnail of the WSI
#     def thumb(self, mask_overlay: bool = False, max_width: int = 1024) -> np.ndarray:
#         w = max_width if self.image.width > max_width else self.image.width
#         h = int(self.image.height * w / self.image.width)
#         image = self.image.read_block(size=(w, h))
#         if mask_overlay and self.mask:
#             return self.mask.overlay(image)
#         return image

import keras


class PNISegmenter:
    def __init__(self, model_path) -> None:
        # self._model = keras.models.load_model(model_path)
        self._model = "TODO"  # TODO

    def segment(self, image, nerves_mask):  # TODO: Multiple patches or just single one?
        # return self._model.predict(image + nerves_mask) # TODO
        return []

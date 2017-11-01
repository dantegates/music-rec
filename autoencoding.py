import keras.models
from keras.models import Model


class AutoEncoder:
    def __init__(self, model):
        self._model = model
        self._encoder = self._extract_encoder(model)

    @classmethod
    def from_h5(cls, filepath):
        model = keras.models.load_model(filepath)
        return cls(model)

    @staticmethod
    def _extract_encoder(model):
        idx_code_layer = (len(model.layers) // 2) - 1
        encoder = Model(inputs=model.input,
            output=model.layers[idx_code_layer].output)
        return encoder

    def predict(self, X, *args, **kwargs):
        return self._model.predict(X, *args, **kwargs)

    def encode(self, X, *args, **kwargs):
        return self._encoder.predict(X, *args, **kwargs)

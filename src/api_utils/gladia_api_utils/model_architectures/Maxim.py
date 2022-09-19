from importlib import import_module

import numpy as np
from PIL import Image


class Maxim:
    def __init__(self, task, checkpoint) -> None:
        from flax.core import freeze
        from maxim.run_eval import (
            _MODEL_CONFIGS,
            _MODEL_FILENAME,
            _MODEL_VARIANT_DICT,
            get_params,
            make_shape_even,
            mod_padding_symmetric,
        )
        from ml_collections import ConfigDict

        model_mod = import_module(f"maxim.models.{_MODEL_FILENAME}")

        model_configs = ConfigDict(_MODEL_CONFIGS)
        model_configs.variant = _MODEL_VARIANT_DICT[task]

        self.model = model_mod.Model(**model_configs)
        self.params = get_params(checkpoint)

        self.mod_padding_symmetric = mod_padding_symmetric
        self.make_shape_even = make_shape_even
        self.freeze = freeze

    def __call__(self, image):
        input_img = np.asarray(image.convert("RGB"), np.float32) / 255.0

        # Padding images to have even shapes
        height, width = input_img.shape[0], input_img.shape[1]
        input_img = self.make_shape_even(input_img)
        height_even, width_even = input_img.shape[0], input_img.shape[1]

        # padding images to be multiplies of 64
        input_img = self.mod_padding_symmetric(input_img, factor=64)
        input_img = np.expand_dims(input_img, axis=0)

        # handle multi-stage outputs, obtain the last scale output of last stage
        preds = self.model.apply({"params": self.freeze(self.params)}, input_img)
        if isinstance(preds, list):
            preds = preds[-1]
            if isinstance(preds, list):
                preds = preds[-1]

        preds = np.array(preds[0], np.float32)

        # unpad images to get the original resolution
        new_height, new_width = preds.shape[0], preds.shape[1]
        h_start = new_height // 2 - height_even // 2
        h_end = h_start + height
        w_start = new_width // 2 - width_even // 2
        w_end = w_start + width
        preds = preds[h_start:h_end, w_start:w_end, :]

        return Image.fromarray(
            np.array((np.clip(preds, 0.0, 1.0) * 255.0).astype(np.uint8))
        )

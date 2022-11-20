# inspired by https://github.com/Sanster/lama-cleaner
from email.mime import image
from logging import getLogger

import numpy as np
from gladia_api_utils.inpainting_helper.model_manager import ModelManager
from gladia_api_utils.inpainting_helper.predictor import inpaint
from gladia_api_utils.inpainting_helper.schema import Config

logger = getLogger(__name__)


def predict(original_image: bytes, mask_image: bytes):
    """
    Erase (inpaint) the mask from the given image.

    Args:
        original_image (bytes): the original image to inpaint (erase) the mask from
        mask_image (bytes): the mask used to inpaint (erase) from the original image

    Returns:
        bytes: the inpainted (erased) image
    """

    config = Config(
        ldm_steps=25,
        ldm_sampler="plms",
        hd_strategy="Crop",
        zits_wireframe=True,
        hd_strategy_crop_margin=128,
        hd_strategy_crop_trigger_size=1024,
        hd_strategy_resize_limit=1024,
    )

    model = ModelManager(name="zits")

    image = inpaint(original_image, mask_image, model, config)

    return image

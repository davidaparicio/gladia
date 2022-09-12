from logging import getLogger

import cv2
import numpy as np
import torch

from .helper import load_img, numpy_to_bytes, resize_max_size
from .model_manager import ModelManager
from .schema import Config

logger = getLogger(__name__)


def inpaint(
    original_image: bytes, mask_image: bytes, model: ModelManager, config: Config
) -> bytes:
    image, alpha_channel = load_img(original_image)

    interpolation = cv2.INTER_CUBIC
    size_limit = max(image.shape)

    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    mask, _ = load_img(mask_image, gray=True)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    res_np_img = model(image, mask, config)

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    image = numpy_to_bytes(res_np_img, "png")

    torch.cuda.empty_cache()

    return image

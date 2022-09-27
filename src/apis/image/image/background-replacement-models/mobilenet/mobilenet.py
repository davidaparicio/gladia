from logging import getLogger

import numpy as np
from gladia_api_utils.io import _open
from PIL import Image

from apis.image.image.background_removal_models.mobilenet.mobilenet import (
    predict as background_removal_predict,
)

logger = getLogger(__name__)


def predict(original_image: bytes, background_image: bytes, alignment: str) -> Image:
    """
    Call the model to return the image and replace the background with the background image

    Args:
        original_image (bytes): Image to replace the background from
        background_image (bytes): Image the background will be replaced with
        alignment (str): insertion position type

    Returns:
        Image: Image with the background replaced
    """

    frontImage = background_removal_predict(image=original_image)

    background = _open(background_image)

    # Convert image to RGBA
    frontImage = frontImage.convert("RGBA")

    # Convert image to RGBA
    background = background.convert("RGBA")

    if alignment == "left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the center
        height = (background.height - frontImage.height) // 2

    elif alignment == "right":
        # Calculate width to be at the right
        width = background.width - frontImage.width

        # Calculate height to be at the center
        height = (background.height - frontImage.height) // 2

    elif alignment == "top" or alignment == "top-center":
        # Calculate width to be at the center
        width = (background.width - frontImage.width) // 2

        # Calculate height to be at the top
        height = 0

    elif alignment == "bottom" or alignment == "bottom-center":
        # Calculate width to be at the center
        width = (background.width - frontImage.width) // 2

        # Calculate height to be at the bottom
        height = background.height - frontImage.height

    elif alignment == "top-left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the top
        height = 0

    elif alignment == "top-right":
        # Calculate width to be at the right
        width = background.width - frontImage.width

        # Calculate height to be at the top
        height = 0

    elif alignment == "bottom-left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the bottom
        height = background.height - frontImage.height

    elif alignment == "bottom-right":
        # Calculate width to be at the right
        width = background.width - frontImage.width

        # Calculate height to be at the bottom
        height = background.height - frontImage.height

    elif alignment == "cropped":
        # Calculate width to be at the center
        width = (background.width - frontImage.width) // 2

        # Calculate height to be at the center
        height = (background.height - frontImage.height) // 2

        # Crop the background
        background = background.crop(
            (width, height, width + frontImage.width, height + frontImage.height)
        )
    else:
        # else center
        # Calculate width to be at the center
        width = (background.width - frontImage.width) // 2

        # Calculate height to be at the center
        height = (background.height - frontImage.height) // 2

    # Paste the image on the background
    background.paste(frontImage, (width, height), frontImage)

    return background

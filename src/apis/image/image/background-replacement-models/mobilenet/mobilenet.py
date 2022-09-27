from io import BytesIO
from PIL import Image
import numpy as np
from gladia_api_utils.io import _open
from apis.image.image.background_removal_models.mobilenet.mobilenet import predict as background_removal_predict

from logging import getLogger

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

    front_image = background_removal_predict(image=original_image)

    background = _open(background_image)

    # Convert image to RGBA
    front_image = front_image.convert("RGBA")
    
    # Convert image to RGBA
    background = background.convert("RGBA")

    if alignment == "left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the center
        height = (background.height - front_image.height) // 2

    elif alignment == "right":
        # Calculate width to be at the right
        width = background.width - front_image.width

        # Calculate height to be at the center
        height = (background.height - front_image.height) // 2

    elif alignment == "top" or alignment == "top-center":
        # Calculate width to be at the center
        width = (background.width - front_image.width) // 2

        # Calculate height to be at the top
        height = 0

    elif alignment == "bottom" or alignment == "bottom-center":
        # Calculate width to be at the center
        width = (background.width - front_image.width) // 2

        # Calculate height to be at the bottom
        height = background.height - front_image.height

    elif alignment == "top-left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the top
        height = 0

    elif alignment == "top-right":
        # Calculate width to be at the right
        width = background.width - front_image.width

        # Calculate height to be at the top
        height = 0

    elif alignment == "bottom-left":
        # Calculate width to be at the left
        width = 0

        # Calculate height to be at the bottom
        height = background.height - front_image.height

    elif alignment == "bottom-right":
        # Calculate width to be at the right
        width = background.width - front_image.width

        # Calculate height to be at the bottom
        height = background.height - front_image.height

    elif alignment == "cropped":
        # Calculate width to be at the center
        width = (background.width - front_image.width) // 2

        # Calculate height to be at the center
        height = (background.height - front_image.height) // 2

        # Crop the background
        background = background.crop((width, height, width + front_image.width, height + frontImage.height))
    else:
        # else center
        # Calculate width to be at the center
        width = (background.width - front_image.width) // 2

        # Calculate height to be at the center
        height = (background.height - front_image.height) // 2


    # Paste the image on the background
    background.paste(front_image, (width, height), front_image)
    
    return background
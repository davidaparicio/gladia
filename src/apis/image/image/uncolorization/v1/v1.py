from gladia_api_utils.io import _open
from PIL import Image


def predict(image: bytes) -> Image:
    """
    Take an image as input and return it as grayscale

    Args:
        image (bytes): image to convert to grayscale

    Returns:
        Image: grayscale image
    """

    return _open(image).convert("L")

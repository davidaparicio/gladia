from gladia_api_utils.model_management import download_model
from PIL import Image

from apis.image.image.background_removal_models.OnnxBackgroundRemoval import (
    OnnxBackgroundRemoval,
)

MODEL_PATH = download_model(
    url="https://huggingface.co/Gladiaio/databuzzword_mobile-net_onnx/resolve/main/databuzzword_mobile-net_onnx.onnx",
    output_path="mobile-net_onnx.onnx",
    uncompress_after_download=False,
)

onnx_bg_remover = OnnxBackgroundRemoval(model_path=MODEL_PATH)


def predict(image: bytes) -> Image:
    """
    Call the model to return the image without its background

    Args:
        image (bytes): Image to remove the background from

    Returns:
        Image: Image without its background
    """

    return onnx_bg_remover.remove_bg(image)

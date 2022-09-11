from typing import Tuple

import onnxruntime as ort
from gladia_api_utils.image_management import draw_segment
from gladia_api_utils.io import _open
from gladia_api_utils.model_management import download_model
from numpy import asarray as as_nparray
from numpy import ndarray
from PIL import Image

class OnnxBackgroundRemoval(object):
    """
    Call the model to return the image without its background
    """
    def __init__(self, model_path: str, model_input_size: int=513, model_input_tensor_name: str="ImageTensor:0") -> None:
        """
        Constructor for the BackgroundRemoval class

        Args:
            model_path (str): Path to the model
            model_input_size (int, optional): Size of the input of the model. Defaults to 513.
            model_input_tensor_name (str, optional): Name of the input tensor of the model. Defaults to "ImageTensor:0".

        Returns:
            None
        """
        
        self.model_path = model_path
        self.model_input_size = model_input_size
        self.model_input_tensor_name = model_input_tensor_name


    def remove_bg(self, image: bytes) -> Tuple[Image.Image, ndarray]:
        """
        Call the model to return the image without its background

        Args:
            image (bytes): Image to remove the background from

        Returns:
            Tuple[Image.Image, ndarray]: Image without its background and the segmentation map
        """
        image = _open(image).convert("RGB")

        width, height = image.size
        resize_ratio = 1.0 * self.model_input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert("RGB").resize(target_size, Image.ANTIALIAS)

        ort_sess = ort.InferenceSession(self.model_path)
        seg_map = ort_sess.run(None, {self.model_input_tensor_name: [as_nparray(resized_image)]})[0][0]

        img = draw_segment(resized_image, seg_map)

        return img

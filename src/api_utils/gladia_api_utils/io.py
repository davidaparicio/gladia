import io
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .file_management import get_buffer_category, get_buffer_type, get_mime_category

logger = getLogger(__name__)


def _open(input) -> Any:
    """
    convert input to infer, numpy, PIL image, binary, pdf

    Args:
        input (Any): Input to convert

    Returns:
        Any: Converted input
    """

    output = None

    if isinstance(input, io.BytesIO):
        logger.debug("Converting io.BytesIO to bytes object")
        buffer = input.read()

    elif isinstance(input, str):
        if Path(input).is_file():
            with open(input, "rb") as fh:
                buffer = io.BytesIO(fh.read())
                buffer = buffer.getvalue()
        else:
            buffer = input

    else:
        buffer = input

    logger.debug("infere type")
    btype = get_buffer_category(buffer)

    if btype == "image":
        logger.debug("infere type image")
        output = to_pil(buffer)
    elif btype == "flat_structured_data":
        logger.debug("infere type structured data")
        output = to_pandas(buffer)
    else:
        output = buffer

    return output


def to_numpy(buffer: bytes) -> np.ndarray:
    """
    convert a buffer to numpy array

    Args:
        buffer (bytes): buffer to convert

    Returns:
        numpy.ndarray: numpy array representing the buffer
    """

    return np.array(Image.open(io.BytesIO(buffer)))


def to_pil(buffer: bytes) -> Image:
    """
    convert a buffer to PIL image

    Args:
        buffer (bytes): buffer to convert

    Returns:
        Image: PIL image
    """

    data = to_numpy(buffer)

    return Image.fromarray(np.uint8(data))


def np_to_img_buffer(data: np.ndarray, format: str = "PNG"):
    """
    convert numpy array to image buffer

    Args:
        data (numpy.ndarray): numpy array to convert
        format (str): image format to convert to

    Returns:
        bytes: image buffer
    """

    buf = io.BytesIO()
    img = Image.fromarray(np.uint8(data))
    img.save(buf, format=format)

    return buf.getvalue()


def np_to_img_pil(data: np.ndarray) -> Image:
    """
    convert numpy array to PIL image

    Args:
        data (numpy.ndarray): numpy array to convert

    Returns:
        Image: PIL image
    """

    return Image.fromarray(np.uint8(data))


def to_pandas(buffer: bytes) -> pd.DataFrame:
    """
    convert a buffer to pandas dataframe

    Args:
        buffer (bytes): buffer to convert

    Returns:
        pandas.DataFrame: dataframe
    """

    buffer_mime_type = get_buffer_type(buffer)
    get_buffer_category = get_mime_category(buffer_mime_type)
    output = None

    if buffer_mime_type == "text/csv":
        output = pd.read_csv(buffer)

    elif buffer_mime_type == "json":
        output = pd.read_json(buffer)

    elif get_buffer_category == "excel":
        output = pd.read_json(buffer)

    elif get_buffer_category == "web_content":
        output = pd.read_html(buffer)

    elif buffer_mime_type in ["hdf5", "orc", "parquet", "sas", "spss", "pickle"]:
        error_message = f"Type {buffer_mime_type} is not implemented yet."

        logger.error(error_message)

        raise RuntimeError(error_message)

    return output

import re
from typing import Dict, List, Union

import cv2
import numpy as np
import pytesseract
from gladia_api_utils.io import _open


def predict(image: bytes, source_language: str) -> Dict[str, Union[str, List[str]]]:
    """
    Call the tesseract ocr and return the text detected in the image

    Args:
        image (bytes): The image to be processed
        source_language (str): The language of the image (unused)

    Returns:
        Dict[str, str]: The text detected in the image by the ocr
    """

    del source_language

    image = _open(image).convert("RGB")

    np_image = np.array(image)

    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    gray_thresh = cv2.threshold(
        src=gray_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(gray_thresh)

    out = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)

    result = out.strip()

    return {"prediction": result, "prediction_raw": result.split("\n")}

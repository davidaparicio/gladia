import os

from typing import Dict, List, Union

import textract
from gladia_api_utils.file_management import input_to_files, get_file_extension

from logging import getLogger

from cleantext import clean

logger = getLogger(__name__)

@input_to_files
def predict(image: bytes, source_language: str) -> Dict[str, Union[str, List[str]]]:
    """
    Call the tesseract ocr and return the text detected in the image

    Args:
        image (bytes): The image or file to be processed actually accepts
            csv / doc / docx / eml / epub / gif / jpg / jpeg / json / mp3 / msg /
            odt / ogg / pdf / png / ppt / pptx / ps / rtf / tif / tiff /
            txt / wav / xls / xlsx
        source_language (str): The language of the image or file (unused)

     Returns:
        Dict[str, str]: The text detected in the image by the ocr
    """

    del source_language

    file = image + "." + get_file_extension(image)
    os.rename(image, file)

    result = textract.process(file).decode("utf-8")

    clean_result = clean(result)


    return {"prediction": clean_result, "prediction_raw": [clean_result.split("\n")]}

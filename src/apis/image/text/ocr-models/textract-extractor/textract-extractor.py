import os
from logging import getLogger
from typing import Dict, List, Union

import textract
from cleantext import clean
from wand.image import Image as wi
from gladia_api_utils.file_management import get_file_extension, input_to_files

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
    raw_prediction = [clean_result]

    # if the result is empty and the file is a pdf, try to extract the text from the pdf
    # by converting it to an image and performing ocr on the image
    if not result and get_file_extension(image) == "pdf":
        raw_prediction = []
        logger.info("No text detected in pdf, trying to convert to image")
        # convert pdf file to image
        pdfFile = wi(filename = file, resolution = 300)
        image = pdfFile.convert('jpeg')

        imageBlobs = []

        for img in image.sequence:
            imgPage = wi(image = img)
            imageBlobs.append(imgPage.make_blob('jpeg'))

        extract = []

        for imgBlob in imageBlobs:
            clean_result += clean(textract.process(file).decode("utf-8"))
            raw_prediction.append(clean_result)
            clean_result += "\n"

        # remove the trailing newline
        clean_result = clean_result[:-1]

    os.unlink(file)

    return {"prediction": clean_result, "prediction_raw": raw_prediction}

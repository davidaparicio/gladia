import io
import json
import os
import pathlib
import re
from logging import getLogger
from typing import Union

import numpy as np
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo
from starlette.responses import StreamingResponse

from .file_management import get_file_type

logger = getLogger(__name__)

png_media_type = "image/png"


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy arrays and other types
    will map numpy types to python types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        else:
            return super(NpEncoder, self).default(obj)


def __convert_pillow_image_response(
    image_response: Image.Image, additional_metadata: dict = dict()
) -> StreamingResponse:
    """
    Convert a Pillow image response to a JSON response

    Args:
        image_response (Image.Image): Pillow image response

    Returns:
        StreamingResponse: FastAPI streaming response for the image
    """
    ioresult = io.BytesIO()

    image_response.save(ioresult, format="png")

    ioresult.seek(0)

    returned_response = StreamingResponse(ioresult, media_type=png_media_type)

    if len(additional_metadata) > 0:
        returned_response.headers["gladia_metadata"] = json.dumps(additional_metadata)

    return returned_response


def __convert_ndarray_response(
    response: np.ndarray, output_type: str
) -> Union[StreamingResponse, JSONResponse]:
    """
    Convert a numpy array into Fastapi response
    returns Streaming response if the ndarray is an image
    returns JSON response if the ndarray is a array

    Args:
        response (np.ndarray): numpy array response
        output_type (str): output type of the response, takes value in {‘image’, ‘text’}

    Returns:
        Union[StreamingResponse, JSONResponse]: FastAPI streaming response for an image or JSON response for a table
    """
    if output_type == "image":
        ioresult = io.BytesIO(response.tobytes())
        ioresult.seek(0)

        return StreamingResponse(ioresult, media_type=png_media_type)

    elif output_type == "text":
        return JSONResponse(content=jsonable_encoder(response.tolist()))

    else:
        logger.warning(
            f"response is numpy array but expected output type {output_type} which is not supported."
        )

        return response


def __convert_bytes_response(response: bytes, output_type: str) -> StreamingResponse:
    """
    Convert a bytes response to a Streaming response

    Args:
        response (bytes): bytes response
        output_type (str): output type of the response, only supported value is `image`

    Returns:
        StreamingResponse: FastAPI streaming response for an image
    """
    ioresult = io.BytesIO(response)
    ioresult.seek(0)

    if output_type == "image":
        return StreamingResponse(ioresult, media_type=png_media_type)

    else:
        logger.warning(
            f"response is bytes but expected output type {output_type} which is not supported."
        )

    return response


def __convert_io_response(response: io.IOBase, output_type: str) -> StreamingResponse:
    """
    Convert a io.IOBase response to a Streaming response

    Args:
        response (io.IOBase): io.IOBase response
        output_type (str): output type of the response image, text, audio, video, etc.

    Returns:
        StreamingResponse: FastAPI streaming response for the output type defined
    """
    response.seek(0)

    if output_type == "image":
        return StreamingResponse(response, media_type=png_media_type)

    else:
        logger.warning(
            f"response is io but expected output type {output_type} which is not supported."
        )

    return response


def __load_json_string_representation(json_string: str) -> Union[dict, str]:
    """
    Load a candidate JSON string, clean it and try to loed it as a dictionary
    if it fails interpret it as a string

    Args:
        json_string (str): JSON string

    Returns:
        Union[dict, str]: dictionary if the JSON string is valid, else the original string
    """
    # I decided to use regex instead of ast.literal_eval
    # for security reason.
    # having regex doesn't interpret while
    # ast.literal_eval will
    # see this proposition:
    # https://stackoverflow.com/questions/39491420/python-jsonexpecting-property-name-enclosed-in-double-quotes
    # which I found very risky
    # J.L
    try:
        p = re.compile("(?<!\\\\)'")
        replace_map = [("\n", "\\n"), ("\\\n", "\\n"), ("\\x0c", "")]
        this_response = p.sub('"', json_string)
        for replacement in replace_map:
            this_response.replace(replacement[0], replacement[1])
        return json.loads(this_response)
    except Exception as e:
        logger.warning(f"Couldn't interpret response returning plain response: {e}")
        try:
            return JSONResponse(
                content={
                    "prediction": str(json_string),
                    "prediction_raw": str(json_string),
                }
            )
        except Exception as e:
            logger.warning(f"Couldn't interpret response returning plain response: {e}")
            return json_string


def __load_file_as_response(file_path: str) -> Union[StreamingResponse, JSONResponse]:
    """
    Load a file as a response. JSON files will be loaded as JSON response, other files will be loaded as Streaming response

    Args:
        file_path (str): path to the file

    Returns:
        Union[StreamingResponse, JSONResponse]: FastAPI streaming response for an image or JSON response for a table
    """

    try:
        return json.load(file_path)
    except Exception:
        file_to_stream = open(file_path, "rb")
        return StreamingResponse(file_to_stream, media_type=get_file_type(file_path))
    finally:
        os.remove(file_path)


def __convert_string_response(
    response: str,
) -> Union[JSONResponse, StreamingResponse, str]:
    """
    Convert a string response to a JSON response

    Args:
        response (str): string response

    Returns:
        JSONResponse: FastAPI JSON response for the input string,
        if the string is not interpretable JSON response the plain string is returned
        if the string is a path to a file, a StreamingResponse is returned

    """
    # if response is a string but not a file path
    # try to load it as a json representation
    # else return it as is
    if not os.path.exists(response):
        return __load_json_string_representation(response)

    # if the string looks like a filepath
    # try to load it as a json
    # else try to stream it
    else:
        try:
            if pathlib.Path(response).is_file():
                return __load_file_as_response(response)
            else:
                return response
        except OSError as os_error:
            logger.warning(f"Couldn't interpret stream: {os_error}")
            return response


def cast_response(
    response, expected_output: dict
) -> Union[StreamingResponse, JSONResponse, str]:
    """Cast model response to the expected output type

    Args:
        response (Any): response of the model
        expected_output (dict): dict describing the expected output

    Returns:
        Union[StreamingResponse, JSONResponse, str]: FastAPI streaming response for an bytes or JSON response for a table or plain string
    """
    if isinstance(response, tuple):
        # if the response is a tuple, it means that the model
        # returned a tuple of predictions and additional metadata
        if list(map(type, response)) == [Image.Image, dict]:
            image, addition_exif = response
            # if the image is a Pillow image
            # convert it to a StreamingResponse
            return __convert_pillow_image_response(image, addition_exif)
        else:
            return JSONResponse(
                content=json.loads(
                    json.dumps(response, cls=NpEncoder, ensure_ascii=False)
                )
            )

    elif isinstance(response, Image.Image):
        # if the response is a pillow image
        # convert it to a streaming response
        return __convert_pillow_image_response(response)

    elif isinstance(response, np.ndarray):
        # if the response is a numpy array
        # check if the output type is an image or a table
        # if it is an image, convert it to a StreamingResponse
        # if it is a table, convert it to a JSONResponse
        return __convert_ndarray_response(response, expected_output["type"])

    elif isinstance(response, (bytes, bytearray)):
        # if the response is a bytes or bytearray
        # convert it to a StreamingResponse
        return __convert_bytes_response(response, expected_output["type"])

    elif isinstance(response, io.IOBase):
        # if the response is a io.IOBase
        # convert it to a StreamingResponse
        return __convert_io_response(response, expected_output["type"])

    elif isinstance(response, (list, dict)):
        # if the response is a list or dict
        # convert it to a JSONResponse
        return JSONResponse(
            json.loads(
                json.dumps(response, cls=NpEncoder, ensure_ascii=False).encode("utf8")
            )
        )

    elif isinstance(response, str):
        # if the response is a string
        # convert it to a JSONResponse if it is a json or a json interpretable string
        # convert it to a StreamingResponse if it is a file path with a bytes representation
        # otherwise return it as is
        return __convert_string_response(response)

    elif isinstance(response, (bool, float)):
        # if the response is a bool or a float
        # convert it to a JSONResponse
        return JSONResponse({"prediction": str(response)})

    elif isinstance(response, int):
        # if the response is an int
        # convert it to a JSONResponse
        return JSONResponse({"prediction": response})

    else:
        logger.warning(
            f"Response type not supported ({type(response)}), returning a stream"
        )

        ioresult = response
        ioresult.seek(0)

        return StreamingResponse(ioresult, media_type=png_media_type)

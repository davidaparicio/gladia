import os
import sys
from logging import getLogger
from typing import List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from torch.hub import download_url_to_file, get_dir

logger = getLogger(__name__)


def get_cache_path_by_url(url: str) -> str:
    """
    Get the path to the cached file for the given url for the model on the torch hub.

    Args:
        url (string): the url of the model

    Returns:
        string: the path to the cached file
    """
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")

    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))

    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)

    return cached_file


def download_model(url: str) -> str:
    """
    Download model from url and save to cache dir.

    Args:
        url (string): the url of the model

    Returns:
        string: the path to the cached file
    """

    cached_file = get_cache_path_by_url(url)

    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)

    return cached_file


def ceil_modulo(x: int, mod: int) -> int:
    """
    Ceil x to the nearest multiple of mod.

    Args:
        x (int): the number to be ceiled
        mod (int): the modulo

    Returns:
        int: the ceiled number
    """
    if x % mod == 0:
        return x

    return (x // mod + 1) * mod


def load_jit_model(url_or_path: str, device: str) -> torch.jit.ScriptModule:
    """
    Load a torch jit model from url or path.

    Args:
        url_or_path (string): the url or path of the model
        device (string): the device to load the model (cpu or cuda)

    Returns:
        torch.jit.ScriptModule: the loaded model
    """

    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path)

    logger.debug(f"Load model from: {model_path}")

    try:
        model = torch.jit.load(model_path).to(device)
    except Exception:
        logger.error(
            f"Failed to load {model_path}, delete model and restart lama-cleaner"
        )

    model.eval()

    return model


def load_model(
    model: torch.nn.Module, url_or_path: str, device: str
) -> torch.nn.Module:
    """
    Load a torch model from url or path and return the model in the torch format on a given device.

    Args:
        model (torch.nn.Module): the model to be loaded
        url_or_path (string): the url or path of the model
        device (string): the device to load the model (cpu or cuda)

    Returns:
        torch.nn.Module: the loaded model on the given device
    """

    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path)

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        logger.debug(f"Load model from: {model_path}")

    except Exception:
        logger.error(
            f"Failed to load {model_path}, delete model and restart lama-cleaner"
        )

    model.eval()
    return model


def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    """
    Convert a numpy array to bytes.

    Args:
        image_numpy (np.ndarray): the numpy array to be converted
        ext (string): the extension of the image to convert to

    Returns:
        bytes: the converted bytes
    """

    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]

    image_bytes = data.tobytes()

    return image_bytes


def load_img(img_bytes, gray: bool = False) -> np.ndarray:
    """
    Load image from bytes.

    Args:
        img_bytes (bytes): the bytes to be loaded
        gray (bool): whether to load the image as gray scale

    Returns:
        np.ndarray: the loaded image
    """

    alpha_channel = None
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    return np_img, alpha_channel


def norm_img(np_img: np.ndarray) -> np.ndarray:
    """
    Normalize image in a range of [0, 1] from a numpy array.

    Args:
        np_img (np.ndarray): the image to be normalized

    Returns:
        np.ndarray: the normalized image
    """

    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]

    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255

    return np_img


def resize_max_size(
    np_img: np.ndarray, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resize image to a given maximum size.

    Args:
        np_img (np.ndarray): the image to be resized
        size_limit (int): the maximum size of the image
        interpolation (Any): the interpolation method in cv2 (cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA) (default: cv2.INTER_CUBIC)

    Returns:
        np.ndarray: the resized image
    """

    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]

    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)

    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
) -> np.ndarray:
    """
    Pad image to a given modulo.

    Args:
        img (np.ndarray): the image to be padded [H, W, C]
        mod (int): the modulo to pad the image
        square (bool): whether to pad the image to a square shape (default: False)
        min_size (Optional[int]): the minimum size of the image (default: None)
        min_size: the minimum size of the image (default: None)

    Returns:
        np.ndarray: the padded image
    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Get the bounding boxes from a mask.

    Args:
        mask (np.ndarray): the mask to get the bounding boxes from with format (h, w, 1) with value 0~255

    Returns:
        List[np.ndarray]: the bounding boxes
    """

    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes

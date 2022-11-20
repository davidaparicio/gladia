import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.filters import gaussian


def convert_image_to_base64(image: Image) -> str:
    """
    Convert image to base64

    Args:
        iimagemg (Image): image to convert

    Returns:
        str: base64 string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/jpeg;base64," + img_str.decode()


def compress_jpg_image(image: Image, path_original: str, size=(1920, 1080)) -> str:
    """
    Convert a given file to JPG file

    Args:
        image (Image): image to convert
        path_original (str): path to original image
        size (tuple): size of the image to convert

    Returns:
        str: path to converted image
    """

    width, height = size

    name = os.path.basename(path_original).split(".")
    first_name = os.path.join(os.path.dirname(path_original), name[0] + ".jpg")

    if image.size[0] > width and image.size[1] > height:
        image.thumbnail(size, Image.ANTIALIAS)
        image.save(first_name, quality=85)
    elif image.size[0] > width:
        wpercent = width / float(image.size[0])
        height = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, height), Image.ANTIALIAS)
        image.save(first_name, quality=85)
    elif image.size[1] > height:
        wpercent = height / float(image.size[1])
        width = int((float(image.size[0]) * float(wpercent)))
        image = image.resize((width, height), Image.ANTIALIAS)
        image.save(first_name, quality=85)
    else:
        image.save(first_name, quality=85)

    return first_name


def convert_to_jpg(path_original: str) -> str:
    """
    Convert a given file to JPG file

    Args:
        path_original (str): path to original image

    Returns:
        str: path to converted image
    """
    img = Image.open(path_original)

    if img.format == "JPEG" or img.format == "BMP":
        image = img.convert("RGB")

    elif img.format == "GIF":
        i = img.convert("RGBA")
        bg = Image.new("RGBA", i.size)
        image = Image.composite(i, bg, i)

    elif img.format == "PNG":
        try:
            image = Image.new("RGB", img.size, (255, 255, 255))
            image.paste(img, img)
        except ValueError:
            image = img.convert("RGB")

    compress_jpg_image(image, path_original)
    img.close()

    return path_original


def blur_image(
    image: Image,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    sigma: int = 1,
    multichannel: bool = True,
):
    """
    Square blur in image

    Args:
        image (Image): image to blur
        x0 (int): x0 coordinate (top left)
        x1 (int): x1 coordinate (bottom right)
        y0 (int): y0 coordinate (top left)
        y1 (int): y1 coordinate (bottom right)
        sigma (int): sigma value for gaussian blur (default: 1)
        multichannel (bool): if True, apply blur to all channels (default: True)

    Returns:
        Image: blurred image
    """

    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    im = image.copy()
    sub_im = im[y0:y1, x0:x1].copy()
    blur_sub_im = gaussian(sub_im, sigma=sigma, multichannel=multichannel)
    blur_sub_im = np.round(255 * blur_sub_im)
    im[y0:y1, x0:x1] = blur_sub_im

    return im


def draw_segment(base_image: Image, image_matrix_representation: np.array) -> Image:
    """
    Draw segment on image

    Args:
        baseImg (Image): image to draw segment on
        matImg (np.array): segment to draw

    Returns:
        Image: image with segment drawn
    """

    width, height = base_image.size
    dummy_image = np.zeros([height, width, 4], dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            color = image_matrix_representation[y, x]
            (r, g, b) = base_image.getpixel((x, y))
            if color == 0:
                dummy_image[y, x, 3] = 0
            else:
                dummy_image[y, x] = [r, g, b, 255]

    img = Image.fromarray(dummy_image)

    return img

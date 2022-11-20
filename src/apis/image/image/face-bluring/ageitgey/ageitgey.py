from io import BytesIO

import face_recognition
from gladia_api_utils.file_management import input_to_files
from gladia_api_utils.image_management import blur_image
from gladia_api_utils.io import _open, np_to_img_pil


@input_to_files
def predict(image: str) -> BytesIO:
    """
    Call the model returning the image with the faces blured

    Args:
        image (str): Path to the Image to blur

    Returns:
        BytesIO: Image with the faces blured
    """

    sigma = 50
    width, height = _open(image).size

    image = face_recognition.load_image_file(image)

    locations = face_recognition.face_locations(image)

    for location in locations:
        (start_y, end_x, end_y, start_x) = location

        image = blur_image(image, start_x, end_x, start_y, end_y, sigma=sigma)

    image = np_to_img_pil(image)

    image.resize((width, height))

    return image

import cv2 as cv
import numpy as np
from gladia_api_utils.io import _open
from gladia_api_utils.model_management import download_model
from keras.models import load_model
from PIL import Image

MODEL_FILE = download_model(
    url="https://huggingface.co/Gladiaio/mlhubber_colorize/resolve/main/model.06-2.5489.hdf5",
    output_path="model.06-2.5489.hdf5",
    uncompress_after_download=False,
)

PTS_IN_HULL_FILE = download_model(
    url="https://huggingface.co/Gladiaio/mlhubber_colorize/resolve/main/pts_in_hull.npy",
    output_path="pts_in_hull.npy",
    uncompress_after_download=False,
)


def predict(image: bytes) -> Image:
    """
    Call the model to return the image colorized

    Args:
        image (bytes): Image to colorize

    Returns:
        Image: Colorized image
    """

    image = _open(image).convert("L")
    image = np.array(image)

    model = load_model(MODEL_FILE)

    T = 0.38
    EPSILONG = 1e-6

    h_in, w_in = 256, 256
    h_out, w_out = h_in // 4, w_in // 4

    img_rows, img_cols = image.shape[:2]

    q_ab = np.load(PTS_IN_HULL_FILE)
    nb_q = q_ab.shape[0]

    L = image
    image = cv.resize(image, (h_in, w_in), cv.INTER_CUBIC)

    x_test = np.empty((1, h_in, w_in, 1), dtype=np.float32)
    x_test[0, :, :, 0] = image / 255.0

    colorized_img = model.predict(x_test)
    colorized_img = colorized_img.reshape((h_out * w_out, nb_q))

    colorized_img = np.exp(np.log(colorized_img + EPSILONG) / T)
    colorized_img = colorized_img / np.sum(colorized_img, 1)[:, np.newaxis]

    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    x_a = np.sum(colorized_img * q_a, 1).reshape((h_out, w_out))
    x_b = np.sum(colorized_img * q_b, 1).reshape((h_out, w_out))

    x_a = cv.resize(x_a, (img_cols, img_rows), cv.INTER_CUBIC)
    x_b = cv.resize(x_b, (img_cols, img_rows), cv.INTER_CUBIC)

    x_a = x_a + 128
    x_b = x_b + 128

    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = L
    out_lab[:, :, 1] = x_a
    out_lab[:, :, 2] = x_b

    out_lab = out_lab.astype(np.uint8)
    out_bgr = cv.cvtColor(out_lab, cv.COLOR_LAB2BGR)

    out_bgr = out_bgr.astype(np.uint8)

    return Image.fromarray(cv.cvtColor(out_bgr, cv.COLOR_BGR2RGB))

from gladia_api_utils.io import _open
from gladia_api_utils.model_architectures import Maxim
from gladia_api_utils.model_management import download_model
from PIL import Image

PARAMS_PATH = download_model(
    url="https://storage.googleapis.com/gresearch/maxim/ckpt/Enhancement/LOL/checkpoint.npz",
    output_path="checkpoint.npz",
    uncompress_after_download=False,
)

model = Maxim(task="Enhancement", checkpoint=PARAMS_PATH)


def predict(image: bytes) -> Image:
    image = _open(image)

    return model(image)

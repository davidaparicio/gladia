from importlib_metadata import version
from gladia_api_utils.TorchvisionModelHelper import TorchvisionModel
from gladia_api_utils.io import _open


def predict(image: bytes, top_k: int = 1) -> [str]:
    img = _open(image)

    model = TorchvisionModel(
        model_name="vgg16", weights="VGG16_Weights", weights_version="IMAGENET1K_V1"
    )
    output = model(img, top_k)

    return output

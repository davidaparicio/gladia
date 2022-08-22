from gladia_api_utils.io import _open
from gladia_api_utils.TorchvisionModelHelper import TorchvisionModel
from importlib_metadata import version


def predict(image: bytes, top_k: int = 1) -> dict:
    img = _open(image)

    model = TorchvisionModel(
        model_name="regnet_y_32gf",
        weights="RegNet_Y_32GF_Weights",
        weights_version="IMAGENET1K_SWAG_E2E_V1",
    )
    output = model(img, top_k)

    return { "prediction": output["prediction"], "prediction_raw": output["prediction_raw"] }

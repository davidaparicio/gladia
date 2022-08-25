from typing import Dict, Union

from gladia_api_utils.io import _open
from gladia_api_utils.TorchvisionModelHelper import TorchvisionModel
from importlib_metadata import version


def predict(image: bytes, top_k: int = 1) -> Dict[str, Union[str, Dict[str, float]]]:
    img = _open(image)

    model = TorchvisionModel(
        model_name="vit_l_16",
        weights="ViT_L_16_Weights",
        weights_version="IMAGENET1K_SWAG_LINEAR_V1",
    )
    output = model(img, top_k)

    return {
        "prediction": output["prediction"],
        "prediction_raw": output["prediction_raw"],
    }

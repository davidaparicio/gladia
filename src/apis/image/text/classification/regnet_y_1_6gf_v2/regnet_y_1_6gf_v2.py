from typing import Dict, Union

from gladia_api_utils.io import _open
from gladia_api_utils.TorchvisionModelHelper import TorchvisionModel
from importlib_metadata import version


def predict(image: bytes, top_k: int = 1) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Return the top_k predictions for the image classifier

    Args:
        image (bytes): Image to predict
        top_k (int): Number of predictions to return (default: 1)

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: Dictionary of the top_k predictions
    """
    img = _open(image)

    model = TorchvisionModel(
        model_name="regnet_y_1_6gf",
        weights="RegNet_Y_1_6GF_Weights",
        weights_version="IMAGENET1K_V2",
    )
    output = model(img, top_k)

    return {
        "prediction": output["prediction"],
        "prediction_raw": output["prediction_raw"],
    }

from typing import Dict, Union

from gladia_api_utils.io import _open
from torchvision.io import read_image
from torchvision.models import MNASNet1_3_Weights, mnasnet1_3


def predict(image: bytes, top_k: int = 1) -> Dict[str, Union[str, Dict[str, float]]]:
    img = _open(image)

    weights = MNASNet1_3_Weights.DEFAULT
    model = mnasnet1_3(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    model_prediction = model(batch).squeeze(0).softmax(0)
    class_id = model_prediction.argmax().item()
    prediction = weights.meta["categories"][class_id]
    prediction_raw = dict(zip(weights.meta["categories"], model_prediction.tolist()))

    return {"prediction": prediction, "prediction_raw": prediction_raw}

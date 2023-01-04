from typing import Dict, List, Union

import numpy as np
import truecase
from transformers import pipeline


def predict(texts: List[str]) -> Dict[str, Union[str, List[float]]]:
    """
    For a given text, predict if it's POSITIVE, NEUTRAL or NEGATIVE

    Args:
        text (str): The text to predict the label for.

    Returns:
        Dict[str, Union[str, List[float]]]: The predicted label and the associated score POSITIVE, NEUTRAL or NEGATIVE.
    """

    classifier = pipeline("zero-shot-classification")

    prediction = []
    prediction_raw = []
    for text in texts:
        pred = classifier(
            truecase.get_true_case(text),
            candidate_labels=["POSITIVE", "NEUTRAL", "NEGATIVE"],
        )
        prediction.append(pred["labels"][np.argmax(pred["scores"])])
        prediction_raw.append(pred)

    return {"prediction": prediction, "prediction_raw": prediction_raw}

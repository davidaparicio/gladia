from typing import Dict, List, Union

import numpy as np
import truecase
from transformers import pipeline


def predict(text: str) -> Dict[str, Union[str, List[float]]]:
    """
    For a given text, predict if it's POSITIVE, NEUTRAL or NEGATIVE

    Args:
        text (str): The text to predict the label for.

    Returns:
        Dict[str, Union[str, List[float]]]: The predicted label and the associated score POSITIVE, NEUTRAL or NEGATIVE.
    """

    classifier = pipeline("zero-shot-classification")
    prediction = classifier(
        truecase.get_true_case(text),
        candidate_labels=["POSITIVE", "NEUTRAL", "NEGATIVE"],
    )

    label = prediction["labels"][np.argmax(prediction["scores"])]

    return {"prediction": label, "prediction_raw": prediction}

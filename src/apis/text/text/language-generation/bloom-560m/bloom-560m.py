from typing import Dict, Union

import truecase
from transformers import pipeline


def predict(text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Generate the continuation of the sentence

    Args:
        text: sentence to continue

    Returns:
        Dict[str, Union[str, Dict[str, str]]]: continuation of the sentence
    """

    generator = pipeline("text-generation", model="bigscience/bloom-560m")

    res = generator(
        truecase.get_true_case(text), max_length=50, do_sample=True, temperature=0.9
    )

    del generator

    prediction = res[0]["generated_text"]

    return {"prediction": prediction, "prediction_raw": res}

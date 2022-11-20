from typing import Dict, Union

import truecase
from transformers import pipeline


def predict(text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Generate the continuation of the sentence

    Args:
        text (str): The sentence to generate the continuation of

    Returns:
        Dict[str, Union[str, Dict[str, str]]]: The continuation of the sentence
    """

    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

    res = generator(
        truecase.get_true_case(text), max_length=50, do_sample=True, temperature=0.9
    )

    del generator

    prediction = res[0]["generated_text"]

    return {"prediction": prediction, "prediction_raw": res}

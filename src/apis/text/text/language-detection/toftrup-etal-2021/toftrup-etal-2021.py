from typing import Dict, Union

from LanguageIdentifier import rank


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    From a given text, return a json scoring the probability of the given text to be of a certain language

    Args:
        text (str): The text to detect the language of

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The language of the text and the probability of the text to be of that language
    """

    prediction_raw = {}

    for lang, score in rank(text):
        prediction_raw[lang] = score

    prediction = max(zip(prediction_raw.values(), prediction_raw.keys()))[1]

    return {"prediction": prediction, "prediction_raw": prediction_raw}

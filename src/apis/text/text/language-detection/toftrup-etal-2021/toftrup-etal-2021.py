from typing import Dict, Union

from LanguageIdentifier import rank

from langcodes import Language


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    From a given text, return a json scoring the probability of the given text to be of a certain language

    Args:
        text (str): The text to detect the language of

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The language of the text and the probability of the text to be of that language in iso639-3 format
    """

    prediction_raw = {}

    for lang, score in rank(text):
        prediction_raw[Language.get(lang).to_alpha3()] = score

    prediction = Language.get(max(zip(prediction_raw.values(), prediction_raw.keys()))[1]).to_alpha3()

    return {"prediction": prediction, "prediction_raw": prediction_raw}

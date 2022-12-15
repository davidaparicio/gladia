from typing import Dict, List, Union

from deepmultilingualpunctuation import PunctuationModel
import os
from logging import getLogger

from gladia_api_utils.model_management import (
    load_spacy_language_model
)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "toftrup-etal-2021",
    os.path.join(
        os.getenv("PATH_TO_GLADIA_SRC", "/app"),
        "apis/text/text/language-detection/toftrup-etal-2021/toftrup-etal-2021.py"
        )
    )

toftrup_etal_2021 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toftrup_etal_2021)


LANGUAGE_PUNCTUATION_MODEL_MAPPING = {
    "eng": "oliverguhr/fullstop-punctuation-multilang-large",
    "deu": "oliverguhr/fullstop-punctuation-multilang-large",
    "fra": "oliverguhr/fullstop-punctuation-multilang-large",
    "ita": "oliverguhr/fullstop-punctuation-multilang-large",
    "nld": "oliverguhr/fullstop-dutch-sonar-punctuation-prediction",
    "cat": "softcatala/fullstop-catalan-punctuation-prediction",
    "others": "kredor/punctuate-all"
}


logger = getLogger(__name__)


def predict(sentence: str) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
    """
    Format the input sentence as a punctuation-restored sentence

    Args:
        sentence (str): The input string to be punctuated
    Returns:
        Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]: The punctuation-restored string
    """

    detected_language = toftrup_etal_2021.predict(sentence)["prediction"]

    if detected_language not in LANGUAGE_PUNCTUATION_MODEL_MAPPING:
        model = PunctuationModel(model=LANGUAGE_PUNCTUATION_MODEL_MAPPING["others"])
    else:
        model = PunctuationModel(model=LANGUAGE_PUNCTUATION_MODEL_MAPPING[detected_language])

    original_sentences = sentence.split(".")
    prediction = ""
    prediction_raw = list()

    for _, original_sentence in enumerate(original_sentences):

        restored_sentence = (model.restore_punctuation(original_sentence) + ".").strip()

        restored_sentence = restored_sentence.replace("...", "<GLADIA_THREE_DOTS_TOKEN>").replace("..", ".").replace(" , ",", ").replace(" ," ,", ").replace(" . ",". ").replace(" .", ". ").replace(" ! ","! ").replace(" !","! ").replace(" ? ","?").replace(" ?","? ").replace(" ; ",";").replace(" ;","; ").replace(" : ",": ").replace(" :",": ").replace("<GLADIA_THREE_DOTS_TOKEN>", "...")

        prediction += restore_capitalization(restored_sentence, detected_language)
        prediction = prediction[0].upper() + prediction[1:]
        prediction_raw.append(
            {
                "detected_language": detected_language,
                "original_sentence": original_sentence,
                "restored_sentence": restored_sentence,
                "scores": model.predict(model.preprocess(sentence)),
            }
        )

    return {"prediction": prediction, "prediction_raw": prediction_raw}


def restore_capitalization(sentence: str, language: str) -> str:
    """
    Restore the capitalization of the input sentence

    Args:
        sentence (str): The input string to be capitalized
        language (str): The language of the input string

    Returns:
        str: The capitalized string
    """
    nlp = load_spacy_language_model(language)

    # Create a spaCy document from the sentence
    doc = nlp(sentence)

    # Iterate over the named entities in the document
    for ent in doc.ents:
        # Check if the entity is a person or an organization
        if ent.label_ == 'PERSON' or ent.label_ == 'ORG' or ent.label_ == 'GPE':
            # If so, capitalize the word
            sentence = sentence.replace(ent.text, ent.text.capitalize())

    # Return the capitalized sentence
    return sentence






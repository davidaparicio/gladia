from typing import Dict, Union

import spacy
import os

try:
    EN_CORE_WEB_LG = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    EN_CORE_WEB_LG = spacy.load("en_core_web_lg")


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Apply NER on the given task and return each token within the sentence associated to its label.

    **Labels**:\n
    O : Outside of a named entity\n
    B-MISC : Beginning of a miscellaneous entity right after another miscellaneous entity\n
    I-MISC : Miscellaneous entity\n
    B-PER : Beginning of a person's name right after another person's name\n
    I-PER : Person's name\n
    B-ORG : Beginning of an organisation right after another organisation\n
    I-ORG : Organisation\n
    B-LOC : Beginning of a location right after another location\n
    I-LOC : Location\n

    :param text: sentence to search the named entities in

    :return: each token within the sentence associated to its label
    """

    document = EN_CORE_WEB_LG(text)

    prediction_raw = []
    for entity in document.ents:
      extraction = {}
      extraction["first_index"] = entity.start_char
      extraction["last_index"] = entity.end_char
      extraction["name"] = entity.label_
      extraction["content"] = entity.text
      prediction_raw.append(extraction)

    return {"prediction": prediction_raw, "prediction_raw": prediction_raw}

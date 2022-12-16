import importlib.util
from typing import Dict, Union
import os
from gladia_api_utils.model_management import load_spacy_language_model
from gladia_api_utils.nlp import explain_ner

spec = importlib.util.spec_from_file_location(
    "toftrup-etal-2021",
    os.path.join(
        os.getenv("PATH_TO_GLADIA_SRC", "/app"),
        "apis/text/text/language-detection/toftrup-etal-2021/toftrup-etal-2021.py",
    ),
)

toftrup_etal_2021 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toftrup_etal_2021)


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Apply NER on the given task and return each token within the sentence associated to its label.

    params:
        text(str): sentence to search the named entities in

    return:
        Dict[str, Union[str, Dict[str, float]]]: each token within the sentence associated to its label
    """
    detected_language = toftrup_etal_2021.predict(text)["prediction"]
    nlp = load_spacy_language_model(detected_language)
    document = nlp(text)

    prediction_raw = []
    prediction = []

    for entity in document.ents:
        extraction = {}
        extraction["content"] = entity.text
        extraction["name"] = entity.label_
        extraction["first_index"] = entity.start_char
        extraction["last_index"] = entity.end_char
        extraction["description"] = explain_ner(entity.label_)

        prediction.append({"text": entity.text, "label": entity.label_})

        prediction_raw.append(extraction)

    return {"prediction": prediction, "prediction_raw": prediction_raw}

from typing import Dict, Union
import spacy


EN_CORE_WEB_LG = spacy.load("en_core_web_lg")


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Apply NER on the given task and return each token within the sentence associated to its label.

    params:
        text(str): sentence to search the named entities in

    return:
        Dict[str, Union[str, Dict[str, float]]]: each token within the sentence associated to its label
    """

    document = EN_CORE_WEB_LG(text)

    prediction_raw = []
    prediction = []
    
    for entity in document.ents:
      extraction = {}
      extraction["first_index"] = entity.start_char
      extraction["last_index"] = entity.end_char
      extraction["name"] = entity.label_
      extraction["content"] = entity.text

      prediction.append({"text": entity.text, "label": entity.label_})

      prediction_raw.append(extraction)

    return {"prediction": prediction, "prediction_raw": prediction_raw}

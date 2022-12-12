from typing import Dict, Union

import mii
from gladia_api_utils.deepspeed_helper import prepare as prepare_deepspeed

MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
DEPLOYMENT_NAME = MODEL_NAME + "-NER-deployment"


def prepare(*_, **__) -> None:
    prepare_deepspeed()

    mii.deploy(
        task="token-classification",
        model=MODEL_NAME,
        deployment_name=DEPLOYMENT_NAME,
        mii_config={
            "tensor_parallel": 1,
            "port_number": 50051,
        },
    )


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

    generator = mii.mii_query_handle(DEPLOYMENT_NAME)

    result = generator.query(
        {
            "query": text,
        }
    ).response

    return {"prediction": result, "prediction_raw": result}

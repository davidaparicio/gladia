from json import loads as json_loads
from typing import Dict, Union

import mii
from gladia_api_utils.deepspeed_helper import prepare as prepare_deepspeed

MODEL_NAME = "jb2k/bert-base-multilingual-cased-language-detection"
DEPLOYMENT_NAME = MODEL_NAME + "-LanguageDetection-deployment"


LABELS = {
    "LABEL_0": "arb",
    "LABEL_1": "baq",
    "LABEL_2": "bre",
    "LABEL_3": "cat",
    "LABEL_4": "chi",
    "LABEL_5": "chi",
    "LABEL_6": "chi",
    "LABEL_7": "chv",
    "LABEL_8": "cze",
    "LABEL_9": "div",
    "LABEL_10": "dum",
    "LABEL_11": "eng",
    "LABEL_12": "epo",
    "LABEL_13": "est",
    "LABEL_14": "fre",
    "LABEL_15": "fry",
    "LABEL_16": "geo",
    "LABEL_17": "get",
    "LABEL_18": "grc",
    "LABEL_19": "cnh",
    "LABEL_20": "ind",
    "LABEL_21": "ina",
    "LABEL_22": "ita",
    "LABEL_23": "jpn",
    "LABEL_24": "kab",
    "LABEL_25": "kin",
    "LABEL_26": "kir",
    "LABEL_27": "lav",
    "LABEL_28": "mlt",
    "LABEL_29": "mon",
    "LABEL_30": "peo",
    "LABEL_31": "pol",
    "LABEL_32": "prt",
    "LABEL_33": "ron",
    "LABEL_34": "roh",
    "LABEL_35": "rus",
    "LABEL_36": "sah",
    "LABEL_37": "slv",
    "LABEL_38": "spa",
    "LABEL_39": "swe",
    "LABEL_40": "tam",
    "LABEL_41": "tat",
    "LABEL_42": "tur",
    "LABEL_43": "ukr",
    "LABEL_44": "wel",
}


def prepare(*_, **__) -> None:

    prepare_deepspeed()

    mii.deploy(
        task="text-classification",
        model=MODEL_NAME,
        deployment_name=DEPLOYMENT_NAME,
        mii_config={"tensor_parallel": 1, "port_number": 50052},
    )


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:

    generator = mii.mii_query_handle(DEPLOYMENT_NAME)

    result = generator.query(
        {
            "query": text,
        }
    ).response

    max_language = ""
    max_score = 0.0

    results = json_loads(result.replace("'", '"'))[0]
    prediction_raw = dict()

    for result in results:
        language = LABELS[result["label"]]
        prediction_raw[language] = result["score"]
        if result["score"] > max_score:
            max_language = language
            max_score = result["score"]

    return {"prediction": max_language, "prediction_raw": prediction_raw}

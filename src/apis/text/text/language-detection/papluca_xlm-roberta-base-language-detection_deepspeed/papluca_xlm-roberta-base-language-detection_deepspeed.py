from json import loads as json_loads
from typing import Dict, Union

import mii
from gladia_api_utils.deepspeed_helper import warm_up as warm_up_deepspeed

MODEL_NAME = "jb2k/bert-base-multilingual-cased-language-detection"
DEPLOYMENT_NAME = MODEL_NAME + "-LanguageDetection-deployment"


def warm_up(*_, **__) -> None:

    warm_up_deepspeed()

    mii.deploy(
        task="text-classification",
        model=MODEL_NAME,
        deployment_name=DEPLOYMENT_NAME,
        mii_config={"tensor_parallel": 1, "port_number": 50052},
    )


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:

    LABELS = {
        "LABEL_0": "Arabic",
        "LABEL_1": "Basque",
        "LABEL_2": "Breton",
        "LABEL_3": "Catalan",
        "LABEL_4": "Chinese_China",
        "LABEL_5": "Chinese_Hongkong",
        "LABEL_6": "Chinese_Taiwan",
        "LABEL_7": "Chuvash",
        "LABEL_8": "Czech",
        "LABEL_9": "Dhivehi",
        "LABEL_10": "Dutch",
        "LABEL_11": "English",
        "LABEL_12": "Esperanto",
        "LABEL_13": "Estonian",
        "LABEL_14": "French",
        "LABEL_15": "Frisian",
        "LABEL_16": "Georgian",
        "LABEL_17": "German",
        "LABEL_18": "Greek",
        "LABEL_19": "Hakha_Chin",
        "LABEL_20": "Indonesian",
        "LABEL_21": "Interlingua",
        "LABEL_22": "Italian",
        "LABEL_23": "Japanese",
        "LABEL_24": "Kabyle",
        "LABEL_25": "Kinyarwanda",
        "LABEL_26": "Kyrgyz",
        "LABEL_27": "Latvian",
        "LABEL_28": "Maltese",
        "LABEL_29": "Mongolian",
        "LABEL_30": "Persian",
        "LABEL_31": "Polish",
        "LABEL_40": "Portuguese",
        "LABEL_41": "Romanian",
        "LABEL_42": "Romansh_Sursilvan",
        "LABEL_43": "Russian",
        "LABEL_44": "Sakha",
        "LABEL_45": "Slovenian",
        "LABEL_46": "Spanish",
        "LABEL_47": "Swedish",
        "LABEL_48": "Tamil",
        "LABEL_49": "Tatar",
        "LABEL_50": "Turkish",
        "LABEL_51": "Ukranian",
        "LABEL_52": "Welsh",
    }

    generator = mii.mii_query_handle(DEPLOYMENT_NAME)

    result = generator.query(
        {
            "query": text,
        }
    ).response

    language = ""
    max_score = 0.0

    for p in json_loads(result.replace("'", '"'))[0]:
        if p["score"] > max_score:
            max_score = p["score"]
            language = p["label"]

    return {"prediction": LABELS[language], "prediction_raw": result}

from typing import Dict, List

import truecase
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def predict(texts: List[str]) -> Dict[str, str]:
    """
    From a given sentence, return the emotion detected in it with the following possible emotions:
        - sadness
        - joy
        - love
        - anger
        - fear
        - surprise

    Args:
        text (str): The sentence to analyze.

    Returns:
        Dict[str, str]: The detected emotion (sadness, joy, love, anger, fear, surprise).
    """

    model_name = "mrm8488/t5-base-finetuned-emotion"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    result = []
    for text in texts:
        input_ids = tokenizer.encode(truecase.get_true_case(text), return_tensors="pt")

        outputs = model.generate(input_ids)

        decoded = tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        result.append(decoded)

    del model
    del tokenizer

    return {"prediction": result, "prediction_raw": result}

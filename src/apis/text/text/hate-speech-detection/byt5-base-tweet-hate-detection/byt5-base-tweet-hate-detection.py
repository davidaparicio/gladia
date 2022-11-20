from typing import Dict, Union

import truecase
from torch import cuda, device
from transformers import ByT5Tokenizer, T5ForConditionalGeneration


def predict(text: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Detect hate from a given text

    Args:
        text (str): The text to be detect hate in

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The level of hate in the text (normal, hate-speech, offensive)
    """

    device_to_use = device("cuda" if cuda.is_available() else "cpu")

    ckpt = "Narrativa/byt5-base-tweet-hate-detection"

    tokenizer = ByT5Tokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device_to_use)

    inputs = tokenizer(
        truecase.get_true_case(text),
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    input_ids = inputs.input_ids.to(device_to_use)
    attention_mask = inputs.attention_mask.to(device_to_use)

    output = model.generate(input_ids, attention_mask=attention_mask)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "prediction": "normal" if output == "no-hate-speech" else output,
        "prediction_raw": output,
    }

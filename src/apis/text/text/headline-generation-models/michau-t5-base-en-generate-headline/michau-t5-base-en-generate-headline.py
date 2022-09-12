from typing import Dict, Union

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def predict(text: str, max_length: int = 16) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Predict the headline from a given text

    Args:
        text (str): The text to be detect hate in
        max_length (int): The maximum length of the headline

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The predicted headline
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained(
        "Michau/t5-base-en-generate-headline"
    )
    tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
    model = model.to(device)

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=max_length,
        num_beams=3,
        early_stopping=True,
    )

    result = tokenizer.decode(beam_outputs[0]).replace("<pad> ", "").replace("</s>", "")

    return {"prediction": result, "prediction_raw": result}

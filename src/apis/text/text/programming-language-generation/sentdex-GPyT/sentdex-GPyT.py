from typing import Dict, Tuple, Union

import torch
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def generate(
    code: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_length: int = 100,
) -> Tuple:
    """
    Takes input code, replaces newline chars with <N>,
    tokenizes, feeds through model, decodes, then reformat the newlines back in.

    Args:
        code (str): The code to generate the continuation of
        tokenizer (PreTrainedTokenizer): The tokenizer to use
        model (PreTrainedModel): The model to use
        max_length (int): The maximum length of the generated code

    Returns:
        str: The generated code
    """

    new_line_token = "<N>"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    converted = code.replace("\n", new_line_token)

    tokenized = tokenizer.encode(converted, return_tensors="pt").to(device)
    resp = model.generate(tokenized, max_length=max_length).to(device)

    decoded = tokenizer.decode(resp[0])
    reformatted = decoded.replace(new_line_token, "\n")

    return reformatted, decoded


def predict(code_snippet: str) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Generate the continuation of the provided `code_snippet`.

    Args:
        code_snippet (str): The code to generate the continuation of

    Returns:
        Dict[str, str]: The generated code
    """

    model_name = "Sentdex/GPyT"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    result, result_raw = generate(code_snippet, tokenizer, model)

    del model
    del tokenizer

    return {"prediction": result, "prediction_raw": result_raw}

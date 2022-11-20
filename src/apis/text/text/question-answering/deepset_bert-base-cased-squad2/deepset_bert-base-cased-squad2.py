from typing import Dict, List, Tuple, Union

import torch
import truecase
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model() -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load `deepset/bert-base-cased-squad2` tokenizer and model.


    :return: tuple containing both the tokenizer and the model
    """

    model_name = "deepset/bert-base-cased-squad2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    return tokenizer, model


def predict(
    context: str, question: str, top_k: int = 1
) -> Dict[str, Union[str, List[Dict[str, Union[str, float, int]]]]]:
    """
    Using the given `context`, answer the provided `question`.

    Note: The score is not provided in the return value.

    Args:
        context (str): The context to answer the question in
        question (str): The question to answer

    Returns:
       Dict[str, Union[str, List[Dict[str, Union[str, float, int]]]]]: The answer to the question
    """

    tokenizer, model = load_model()

    inputs = tokenizer.encode_plus(
        truecase.get_true_case(question),
        truecase.get_true_case(context),
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].tolist()[0]

    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    result = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )

    del tokenizer
    del model

    return {"prediction": result, "prediction_raw": result}

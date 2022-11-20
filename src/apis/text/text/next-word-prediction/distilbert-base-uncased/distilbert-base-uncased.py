from typing import Dict, Union

from transformers import AutoModelForMaskedLM, DistilBertTokenizerFast, FillMaskPipeline


def predict(sentence: str, top_k: int = 3) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    For a given sentence, predict the next word.

    Args:
        sentence (str): The sentence to predict the next word from.

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The next word predicted and score from the sentence.
    """

    model_checkpoint = "distilbert-base-uncased"

    pipeline = FillMaskPipeline(
        model=AutoModelForMaskedLM.from_pretrained(model_checkpoint),
        tokenizer=DistilBertTokenizerFast.from_pretrained(model_checkpoint),
    )

    answers = pipeline(f"{sentence} [MASK]", top_k=top_k)

    return {
        "prediction": answers[0]["token_str"],
        "prediction_raw": [
            (answer["token_str"], answer["score"]) for answer in answers
        ],
    }

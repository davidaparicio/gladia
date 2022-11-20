from typing import Dict, Union

from happytransformer import HappyWordPrediction


def predict(sentence: str, top_k: int = 3) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    For a given sentence, predict the next word.

    Args:
        sentence (str): The sentence to predict the next word from.

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The next word predicted and score from the sentence.
    """

    happy_wp = HappyWordPrediction(model_type="ALBERT", model_name="albert-base-v2")

    result = happy_wp.predict_mask(f"{sentence} [MASK]", top_k=top_k)

    prediction_raw = {word.token: word.score for word in result}
    prediction = result[0].token

    del happy_wp

    return {"prediction": prediction, "prediction_raw": prediction_raw}

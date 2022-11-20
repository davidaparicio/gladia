from typing import Dict, Union


def predict(sentence: str, top_k: int = 3) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    For a given sentence, predict the next word.

    :param sentence: sentence to continue
    :return: word predicted and score
    """

    from happytransformer import HappyWordPrediction

    happy_wp = HappyWordPrediction("ROBERTA", "roberta-base")

    result = happy_wp.predict_mask(f"{sentence} [MASK]", top_k=top_k)

    prediction_raw = {word.token: word.score for word in result}
    prediction = result[0].token

    del happy_wp

    return {"prediction": prediction, "prediction_raw": prediction_raw}

from typing import Dict, List, Union

import truecase
from happytransformer import HappyTextClassification


def predict(
    texts: List[str],
) -> Dict[str, Union[str, Dict[str, Union[str, List[str], List[float]]]]]:
    """
    For a given text, predict if it's POSITIVE or NEGATIVE

    Args:
        text (str): The text to predict the label for.

    Returns:
        Dict[str, Union[str, Dict[str, Union[str, List[str], List[float]]]]]: The predicted label and the associated score POSITIVE or NEGATIVE.
    """

    happy_tc = HappyTextClassification(
        "DISTILBERT", "distilbert-base-uncased", num_labels=2
    )

    prediction = []
    prediction_raw = []
    for text in texts:
        result = happy_tc.classify_text(truecase.get_true_case(text))
        prediction.append("POSITIVE" if result.label == "LABEL_0" else "NEGATIVE")
        prediction_raw.append({"label": result.label, "score": result.score})

    return {"prediction": prediction, "prediction_raw": prediction_raw}

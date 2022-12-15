from typing import Dict, List, Union
from fastpunct import FastPunct
import truecase



def predict(sentence: str) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
    """
    Format the input sentence as a punctuation-restored sentence

    Args:
        sentence (str): _description_

    Returns:
        Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]: _description_
    """

    fastpunct = FastPunct()

    original_sentences = sentence.split(".")
    prediction = ""
    prediction_raw = list()

    for _, original_sentence in enumerate(original_sentences):

        restored_sentence = truecase.get_true_case(
            fastpunct.punct(original_sentence.strip()).strip()
        )

        prediction += restored_sentence

        prediction_raw.append(
            {
                "original_sentence": original_sentence,
                "restored_sentence": restored_sentence,
                "scores": 1,
            }
        )

    return {"prediction": prediction, "prediction_raw": prediction_raw}


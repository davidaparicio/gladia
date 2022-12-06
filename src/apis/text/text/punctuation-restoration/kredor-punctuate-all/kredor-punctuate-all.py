from typing import Dict, List, Union

import truecase
from deepmultilingualpunctuation import PunctuationModel

model = PunctuationModel(model="kredor/punctuate-all")


def predict(sentence: str) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
    """
    Format the input sentence as a punctuation-restored sentence

    Args:
        sentence (str): The input string to be punctuated
    Returns:
        Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]: The punctuation-restored string
    """

    original_sentences = sentence.split(".")
    prediction = ""
    prediction_raw = list()

    for _, original_sentence in enumerate(original_sentences):
        restored_sentence = truecase.get_true_case(
            (model.restore_punctuation(original_sentence) + ".").strip()
        )
        prediction += restored_sentence
        prediction_raw.append(
            {
                "original_sentence": original_sentence,
                "restored_sentence": restored_sentence,
                "scores": model.predict(model.preprocess(sentence)),
            }
        )

    return {"prediction": prediction, "prediction_raw": prediction_raw}

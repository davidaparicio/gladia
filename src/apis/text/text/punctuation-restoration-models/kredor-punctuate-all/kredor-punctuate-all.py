from typing import Dict
from deepmultilingualpunctuation import PunctuationModel

model = PunctuationModel(model = "kredor/punctuate-all")

def predict(sentence: str) -> Dict[str, str]:
    """
    Format the input sentence as a punctuation-restored sentence

    Args:
        sentence (str): The input string to be punctuated
    Returns:
        Dict[str, str]: The punctuation-restored string
    """

    return {"prediction": model.restore_punctuation(sentence), "raw_prediction": model.predict(model.preprocess(sentence))}

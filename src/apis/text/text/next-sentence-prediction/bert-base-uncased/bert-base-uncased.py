from typing import Dict

from happytransformer import HappyNextSentence


def predict(sentence_1: str, sentence_2: str) -> Dict[str, float]:
    """
    Tell for a given sencence_2, whether or not it follows sentence_1

    Args:
        sentence_1 (str): Preceding sentence
        sentence_2 (str): Sentence to check

    Returns:
        Dict[str, float]: confidence score, >= 0.5 if sentence_2 follows sentence_1, else < 0
    """

    happy_ns = HappyNextSentence("BERT", "bert-base-uncased")

    result = happy_ns.predict_next_sentence(sentence_1, sentence_2)

    return {"prediction": result, "prediction_raw": result}

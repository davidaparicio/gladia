# inspired from https://huggingface.co/spaces/ml6team/post-processing-summarization/
from typing import Dict

from transformers import pipeline


def predict(
    text: str,
) -> Dict[str, str]:
    """
    Predict the summary of a text

    Args:
      text (str): text to summarize

    Returns:
        Dict[str, str]: summary of the conversation
    """

    summarizer = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")

    chunk_size = int(len(text) // 3.5)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    predictions = []
    predictions_raw = []
    for chunk in chunks:
        prediction = summarizer(chunk)
        predictions.append(prediction[0]["summary_text"])
        predictions_raw.append(prediction)

    return {"prediction": " ".join(predictions), "prediction_raw": predictions_raw}

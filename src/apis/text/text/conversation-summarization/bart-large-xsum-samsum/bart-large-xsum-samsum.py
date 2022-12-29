# inspired from https://huggingface.co/spaces/ml6team/post-processing-summarization/
from typing import Dict

from transformers import pipeline
from transformers import BartTokenizerFast


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

    tok = BartTokenizerFast.from_pretrained("knkarthick/bart-large-xsum-samsum")

    tokenized = tok(text, return_tensors="pt")
    tensor = tokenized["input_ids"]

    chunks = []
    for i in range(0, tensor.shape[1], 1000):
        chunks.append(tok.decode(tensor[0][i : i + 1000]))

    predictions = []
    predictions_raw = []
    for chunk in chunks:
        prediction = summarizer(chunk)
        predictions.append(prediction[0]["summary_text"])
        predictions_raw.append(prediction)

    return {"prediction": " ".join(predictions), "prediction_raw": predictions_raw}

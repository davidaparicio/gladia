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
      source_language (str): language of the text
      min_len (int): minimum length of the summary
      max_len (int): maximum length of the summary

    Returns:
        Dict[str, str]: summary of the conversation
    """

    summarizer = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")

    summary = summarizer(text)

    return {"prediction": summary["summary_text"], "prediction_raw": summary}

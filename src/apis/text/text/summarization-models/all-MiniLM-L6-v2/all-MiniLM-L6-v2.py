# inspired from https://huggingface.co/spaces/ml6team/post-processing-summarization/
import itertools
from typing import Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def get_summarizer_model():
    """
    Returns a summarizer model

    Returns:
      summarizer_model (pipeline): summarizer model
    """
    model_name = "google/pegasus-cnn_dailymail"

    summarizer_model = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    return summarizer_model


def generate_abstractive_summary(
    summarization_model: pipeline,
    text: str,
    type: str,
    min_len: int = 120,
    max_len: int = 512,
    **kwargs
):
    """
    Generates an abstractive summary

    Args:
      summarization_model (pipeline): summarization model
      text (str): text to summarize
      type (str): type of summarization
      min_len (int): minimum length of the summary
      max_len (int): maximum length of the summary
      **kwargs: additional arguments

    Returns:
      summary (str): summary
    """

    text = text.strip().replace("\n", " ")
    if type == "top_p":
        text = summarization_model(
            text,
            min_length=min_len,
            max_length=max_len,
            top_k=50,
            top_p=0.95,
            clean_up_tokenization_spaces=True,
            truncation=True,
            **kwargs
        )
    elif type in ["greedy", "beam"]:
        text = summarization_model(
            text,
            min_length=min_len,
            max_length=max_len,
            clean_up_tokenization_spaces=True,
            truncation=True,
            **kwargs
        )
    elif type == "top_k":
        text = summarization_model(
            text,
            min_length=min_len,
            max_length=max_len,
            top_k=50,
            clean_up_tokenization_spaces=True,
            truncation=True,
            **kwargs
        )

    summary = text[0]["summary_text"].replace("<n>", " ")

    return summary


def predict(
    text: str,
    source_language: str = "eng",
    min_length: int = 120,
    max_length: int = 512,
) -> Dict[str, str]:
    """
    Predict the summary of a text

    Args:
        text (str): text to summarize
        source_language (str): language of the text
        min_len (int): minimum length of the summary
        max_len (int): maximum length of the summary

    Returns:
        Dict[str, str]: summary of the text
    """

    summarization_model = get_summarizer_model()

    result = generate_abstractive_summary(
        summarization_model=summarization_model,
        text=text,
        type="beam",
        min_len=min_length,
        max_len=max_length,
        do_sample=True,
        num_beams=15,
        no_repeat_ngram_size=4,
    )

    return {"prediction": result, "prediction_raw": result}

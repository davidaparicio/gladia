from typing import Dict, Union

import numpy as np
import truecase
from gladia_api_utils.triton_helper import (
    TritonClient,
    check_if_model_needs_to_be_preloaded,
    data_processing,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def predict(text: str, top_k: int = 10) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Extract keywords from a given sentence

    Args:
        text (str): The sentence to extract keywords from

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The keywords extracted from the sentence
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2_inference_onnx"

    triton_client = TritonClient(
        model_name=MODEL_NAME,
        preload_model=check_if_model_needs_to_be_preloaded(MODEL_NAME),
        sub_parts=[
            "paraphrase-multilingual-MiniLM-L12-v2_tokenizer_onnx",
            "paraphrase-multilingual-MiniLM-L12-v2_model_onnx",
        ],
        output_name="output",
    )

    np_sentence = data_processing.text_to_numpy(text=truecase.get_true_case(text))

    triton_client.set_input(name="TEXT", shape=np_sentence.shape, datatype="BYTES")

    output = triton_client(np_sentence, unload_model="false")

    # Inspired by https://github.com/MaartenGr/KeyBERT/blob/7b763ae76ddbca56c0d139dd368ce907cee9dd30/keybert/backend/_sentencetransformers.py
    docs = [text]
    candidates = None

    count = CountVectorizer(
        ngram_range=(1, 1),
        stop_words=None,
        min_df=1,
        vocabulary=candidates,
    ).fit(docs)

    words = count.get_feature_names_out()

    df = count.transform(docs)

    doc_embeddings = np.array(output[0])

    word_embeddings = []

    # NOTE: Triton seems to don't handle manually batched inputs (i.e [-1, input_size])
    for idx, word in enumerate(words):
        np_sentence = data_processing.text_to_numpy(text=word)
        triton_client.set_input(name="TEXT", shape=np_sentence.shape, datatype="BYTES")
        word_embeddings.append(
            triton_client(
                np_sentence,
                unload_model="true" if idx == len(words) - 1 else "false",
            )[0][0]
        )

    word_embeddings = np.array(word_embeddings)

    # Find keywords
    all_keywords = []
    for index, _ in enumerate(docs):

        try:
            # Select embeddings
            candidate_indices = df[index].nonzero()[1]
            candidates = [words[index] for index in candidate_indices]
            candidate_embeddings = word_embeddings[candidate_indices]
            doc_embedding = doc_embeddings.reshape(1, -1)

            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            keywords = [
                (candidates[index], round(float(distances[0][index]), 4))
                for index in distances.argsort()[0][-top_k:]
            ][::-1]

            all_keywords.append(keywords)

        # Capturing empty keywords
        except ValueError:
            all_keywords.append([])

    prediction_raw = {k: v for (k, v) in all_keywords[0]}

    return {"prediction": all_keywords[0][0][0], "prediction_raw": prediction_raw}

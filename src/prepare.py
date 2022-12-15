import os

import nltk


def main():
    nltk_warmup_list = ["punkt"]
    spacy_warmup_list = ["en_core_web_lg"]

    download_nltk_data(nltk_warmup_list)
    download_spacy_model(spacy_warmup_list)


def download_nltk_data(nltk_warmup_list: list) -> None:
    """
    Download nltk data

    Args:
        nltk_warmup_list (list): List of nltk data to download

    Returns:
        None
    """
    for tokenizer in nltk_warmup_list:
        try:
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            nltk.download(tokenizer)


def download_spacy_model(spacy_warmup_list: list) -> None:
    """
    Download spacy model

    Args:
        spacy_warmup_list (list): List of spacy model to download

    Returns:
        None
    """

    for spacy_model in spacy_warmup_list:
        try:
            __import__(spacy_model)
        except ImportError:
            os.system("python -m spacy download {}".format(spacy_model))


if __name__ == "__main__":
    main()

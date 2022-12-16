import json
import os

import nltk


def read_config(config_path: str) -> dict:
    """
    Read config file

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Config file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main():
    config = read_config("config.json")

    nltk_warmup_list = ["punkt"]

    spacy_warmup_list = [model["model"] for model in config["spacy"]["models"].values()]

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

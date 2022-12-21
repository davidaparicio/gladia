import json
import os
import threading

import nltk
import spacy

SPACY_CACHE_DIR = os.getenv("SPACY_CACHE_DIR", "/gladia/spacy/models")
NLTK_DATA = os.getenv("NLTK_DATA", "/gladia/nltk_data")

os.makedirs(SPACY_CACHE_DIR) if not os.path.exists(SPACY_CACHE_DIR) else None
os.makedirs(NLTK_DATA) if not os.path.exists(NLTK_DATA) else None


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
    """
    Perform setup tasks for the application.

    This includes reading the configuration file, downloading necessary data and models for natural language processing
    libraries, and any other setup tasks that need to be done before the application can be used.
    Args:
      None

    Returns:
        None
    """
    config = read_config("models-config.json")

    nltk_warmup_list = [
        model["model"] for model in config["nltk"]["tokenizers"].values()
    ]

    spacy_warmup_list = [
        model["model"] for model in config["spacy"]["models"].values()
    ] + ["en_core_web_lg"]

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

    def _download_data(tokenizer):
        try:
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            status = "download"
            nltk.download(tokenizer, download_dir=NLTK_DATA)
        else:
            status = "cache"
        print("\033[31m" + f"NLTK/{tokenizer} > {status}" + "\033[39m")
        

    threads = []
    for tokenizer in nltk_warmup_list:
        t = threading.Thread(target=_download_data, args=(tokenizer,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def download_spacy_model(spacy_warmup_list: list) -> None:
    """
    Download spacy model

    Args:
        spacy_warmup_list (list): List of spacy model to download

    Returns:
        None
    """

    def _download_model(spacy_model):
        try:
            print(colored(f"Spacy/{spacy_model}: ", 'cyan'), end='')
            nlp = spacy.load(os.path.join(SPACY_CACHE_DIR, spacy_model))
        except OSError:
            status = "download"
            spacy.cli.download(spacy_model)
            nlp = spacy.load(spacy_model)
            nlp.to_disk(os.path.join(SPACY_CACHE_DIR, spacy_model))
        else:
            status = "cache"

        print("\033[31m" + f"NLTK/{spacy_model} > {status}" + "\033[39m")

    threads = []
    for spacy_model in spacy_warmup_list:
        t = threading.Thread(target=_download_model, args=(spacy_model,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

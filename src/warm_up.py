import os

import nltk
import importlib


def main():
    nltk.download("punkt")

    spacy_warmup_list = ["en_core_web_lg"]
    for spacy_model in spacy_warmup_list:
        try:
            importlib.import_module(spacy_model)
        except ModuleNotFoundError:
            try:
                os.system("python -m spacy download {}".format(spacy_model))
            except Exception as e:
                print("Error while downloading spacy model: {}".format(e))


if __name__ == "__main__":
    main()

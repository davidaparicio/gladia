import os

import nltk


def main():
    nltk.download("punkt")

    spacy_warmup_list = ["en_core_web_lg"]
    for spacy_model in spacy_warmup_list:
        try:
            os.system("python -m spacy download {}".format(spacy_model))
        except:
            pass


if __name__ == "__main__":
    main()

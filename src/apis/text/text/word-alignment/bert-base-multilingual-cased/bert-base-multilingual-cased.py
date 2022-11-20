import itertools
from typing import Dict, List, Tuple

import torch
from transformers import BertModel, BertTokenizer, PreTrainedTokenizer


def get_tokens(
    input_string: str, tokenizer: PreTrainedTokenizer
) -> Tuple[List[str], List[List[str]]]:
    """
    Tokenize an input string

    Args:
        input_string (str): string to tokenize
        tokenizer (PreTrainedTokenizer): tokenizer to use

    Returns:
        Tuple[List[str], List[List[str]]]: tuple contained the string splited by space and the tokenized string
    """

    sentence = input_string.strip().split()
    token = [tokenizer.tokenize(word) for word in sentence]

    return sentence, token


def tokens_to_tensor(
    tokens: List[List[str]], tokenizer: PreTrainedTokenizer
) -> torch.Tensor:
    """
    From a given tokenized sentence, return it as a Tensor

    Args:
        tokens (List[List[str]]): tokenized sentence
        tokenizer (PreTrainedTokenizer): tokenizer used to tokenize the text

    Returns:
        torch.Tensor: tensor containing the tokenized sentence
    """

    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]

    prepared_ids = tokenizer.prepare_for_model(
        ids=list(itertools.chain(*ids)),
        return_tensors="pt",
        model_max_length=tokenizer.model_max_length,
        truncation=True,
    )

    return prepared_ids["input_ids"]


def map_tokens_to_words(tokens: List[List[str]]) -> List[int]:
    """
    Associate tokens to their respective words.

    Args:
        tokens (List[List[str]]): tokens to associate to their words

    Returns:
        List[int]: word index for which the token come from
    """

    tok2word_map = []

    for i, word_list in enumerate(tokens):
        tok2word_map += [i] * len(word_list)

    return tok2word_map


def get_words_alignment(
    sentence_src: List[str],
    sentence_tgt: List[str],
    tok2word_map_src: List[int],
    tok2word_map_tgt: List[int],
    softmax_inter: torch.Tensor,
) -> List[dict]:
    """
    Associate words from sentence_src to sentence_tgt

    Args:
        sentence_src (List[str]): sentence to associate from
        sentence_tgt (List[str]): sentence to associate to
        tok2word_map_src (List[int]): word index for which the token come from. Map associating tokens to their words for source sentence
        tok2word_map_tgt (List[int]): word index for which the token come from. Map associating tokens to their words for target sentence
        softmax_inter (torch.Tensor): tensor containing the softmax alignment between words from sentence_src to sentence_tgt telling whether or to not add each word

    Returns:
        List[dict]: word alignment from source to target
    """

    align_words = set()
    for i, j in torch.nonzero(softmax_inter, as_tuple=False):
        align_words.add((tok2word_map_src[i], tok2word_map_tgt[j]))

    output = []
    for i, j in sorted(align_words):
        output.append({"source": sentence_src[i], "target": sentence_tgt[j]})

    return output


def predict(
    input_string_language_1: str, input_string_language_2: str
) -> Dict[str, List[Dict[str, str]]]:
    """
    Associated words from `input_string_language_1` to `input_string_language_2`.

    Args:
        input_string_language_1 (str): string to associate from
        input_string_language_2 (str): string to associate to

    Returns:
        Dict[str, List[Dict[str, str]]]: dictionary containing the word alignment from input_string_language_1 to input_string_language_2
    """

    threshold = 1e-3
    align_layer = 8
    model_name = "bert-base-multilingual-cased"

    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    model.eval()

    sentence_src, tokens_src = get_tokens(input_string_language_1, tokenizer)
    sentence_tgt, tokens_tgt = get_tokens(input_string_language_2, tokenizer)

    tensor_src = tokens_to_tensor(tokens=tokens_src, tokenizer=tokenizer)
    tensor_tgt = tokens_to_tensor(tokens=tokens_tgt, tokenizer=tokenizer)

    tok2word_map_src = map_tokens_to_words(tokens_src)
    tok2word_map_tgt = map_tokens_to_words(tokens_tgt)

    with torch.no_grad():
        out_src = model(tensor_src.unsqueeze(0), output_hidden_states=True)[2][
            align_layer
        ][0, 1:-1]
        out_tgt = model(tensor_tgt.unsqueeze(0), output_hidden_states=True)[2][
            align_layer
        ][0, 1:-1]

    del tensor_src, tensor_tgt

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_inter = (torch.nn.Softmax(dim=-1)(dot_prod) > threshold) * (
        torch.nn.Softmax(dim=-2)(dot_prod) > threshold
    )

    result = get_words_alignment(
        sentence_src=sentence_src,
        sentence_tgt=sentence_tgt,
        tok2word_map_src=tok2word_map_src,
        tok2word_map_tgt=tok2word_map_tgt,
        softmax_inter=softmax_inter,
    )

    return {"prediction": result, "prediction_raw": result}

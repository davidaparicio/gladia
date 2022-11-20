from typing import Dict

from transformers import AutoModelWithLMHead, AutoTokenizer


def predict(sentence: str) -> Dict[str, str]:
    """
    Remove typos from the given string.

    Args:
        sentence (str): The sentence to correct.

    Returns:
        Dict[str, str]: The corrected sentence.
    """

    model_name = "flexudy/t5-base-multi-sentence-doctor"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    input_text = f"repair_sentence: {sentence}</s>"

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(input_ids, max_length=32, num_beams=1)

    sentence = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    del model
    del tokenizer

    return {"prediction": sentence, "prediction_raw": sentence}

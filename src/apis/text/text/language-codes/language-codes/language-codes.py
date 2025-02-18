from typing import Dict, Union

from langcodes import Language, standardize_tag


def predict(
    language_code: str, display_output_language: str = "en"
) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    From a given text, return a json scoring the probability of the given text to be of a certain language

    Args:
        language_code (str): The language code to get informations on
        displa_output_language (str): The language code used to display information about the languge

    Returns:
        Dict[str, Union[str, Dict[str, float]]]: The language of the text and the probability of the text to be of that language in iso639-3 format
    """

    prediction_raw = dict()

    l_code = Language.get(language_code)
    l_code_describe = l_code.describe(display_output_language)
    script = ""

    if "script" in l_code_describe:
        script = l_code["script"]

    territory = ""

    if "territory" in l_code_describe:
        territory = l_code["territory"]

    prediction_raw = {
        "alpha3": l_code.to_alpha3(),
        "alpha2": l_code["language"],
        "local_display_name": l_code.autonym(),
        "standardize_tag": standardize_tag(l_code),
        "display_output_language": l_code_describe["language"],
        "script": script,
        "territory": territory,
        "display_name": l_code.display_name(),
        "speaking_population": l_code.speaking_population(),
        "writing_population": l_code.writing_population(),
    }

    return {"prediction": prediction_raw["alpha3"], "prediction_raw": prediction_raw}

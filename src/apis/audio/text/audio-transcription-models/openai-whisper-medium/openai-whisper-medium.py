from typing import Dict

import whisper
from gladia_api_utils.file_management import input_to_files


@input_to_files
def predict(audio: str, language: str = "en") -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be transcribed.
        language (str): The language of

    Outputs:
        Dict[str, str]: The text transcription of the audio.
    """
    model = whisper.load_model("medium")

    result = model.transcribe(audio)

    return {"prediction": result["text"], "prediction_raw": result["text"]}

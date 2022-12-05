from typing import Dict

import whisper
from gladia_api_utils.file_management import input_to_files


@input_to_files
def predict(
    audio: str, language: str = "en", model_version: str = "tiny"
) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be transcribed.
        language (str): The language of
        model_version (str): The model version to use.

    Outputs:
        Dict[str, str]: The text transcription of the audio.
    """
    model = whisper.load_model(model_version)

    transcribe_options = dict(beam_size=5, best_of=5, without_timestamps=False)
    prediction = model.transcribe(audio, **transcribe_options)

    prediction_raw = list()
    for _, segment in enumerate(prediction["segments"]):
        prediction_raw.append(
            {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        )

    return {"prediction": prediction["text"], "prediction_raw": prediction_raw}

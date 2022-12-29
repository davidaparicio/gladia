from typing import Dict

from gladia_api_utils.CoquiEngineHelper import SpeechToTextEngine
from gladia_api_utils.io import _open


def predict(audio: bytes, language: str = "eng", nb_speakers: int = 0
) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be transcribed.
        language (str): The language of the audio to be transcribed. (default: "eng")
        nb_speakers (int): The number of speakers in the audio. If 0, the number of speakers is automatically detected. (unused)

    Returns:
        Dict[str, str]: The text transcription of the audio.
    """

    # Load app configs and initialize STT model
    engine = SpeechToTextEngine(
        model_uri="english/coqui/v1.0.0-huge-vocab",
        model="model.tflite",
        scorer="huge-vocabulary.scorer",
    )

    audio = _open(audio)

    text = engine.run(audio)

    return {"prediction": text, "prediction_raw": text}

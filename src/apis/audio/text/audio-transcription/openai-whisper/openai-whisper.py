from logging import getLogger
from pathlib import Path
from typing import Dict

import whisper
import yaml
from gladia_api_utils import SECRETS
from gladia_api_utils.file_management import (
    delete_file,
    get_tmp_filename,
    input_to_files,
)
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pydub import AudioSegment

logger = getLogger(__name__)


ERROR_MSG = """Error while loading pipeline: {e}
    Please check your HuggingFace credentials in the environment variables HUGGINGFACE_ACCESS_TOKEN
    Also make sure that you have approved the terms of use for the segmentation and diarization models
    for the HUGGINGFACE_ACCESS_TOKEN related token
    """

DEFAULT_MODEL_VERSION = yaml.safe_load(
    open(str(Path(__file__).parent.parent.joinpath("task.yaml")))
)["default-model-version"]

DEFAULT_MODEL = {
    "version": DEFAULT_MODEL_VERSION,
    "model": None,
}


@input_to_files
def predict(
    audio: str, language: str = "en", nb_speakers: int = 0, model_version: str = "tiny"
) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.
    Args:
        audio (bytes): The bytes audio to be transcribed.
        language (str): The language of the audio to be transcribed. (default: "en")
        nb_speakers (int): The number of speakers in the audio. If 0, the number of speakers is automatically detected.
        model_version (str): The model version to use. (default: "tiny")

    Outputs:
        Dict[str, str]: The text transcription of the audio.
    """

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
        )
    except Exception as e:
        logger.error(ERROR_MSG.format(e=e))

    audio_segment = AudioSegment.from_file(audio)

    tmp_file = get_tmp_filename()
    # unfortunately, pyannote.audio only accepts wav files
    # so we need to convert the audio to wav
    # and then delete the file
    # Bytes are said to be supported but it doesn't work
    audio_segment.export(tmp_file, format="wav")

    try:

        if DEFAULT_MODEL["version"] != model_version:
            model = whisper.load_model(model_version)

        else:
            if not DEFAULT_MODEL["model"]:
                DEFAULT_MODEL["model"] = whisper.load_model(DEFAULT_MODEL_VERSION)

            model = DEFAULT_MODEL["model"]

        asr_result = model.transcribe(tmp_file)

        if nb_speakers > 0:
            diarization_result = pipeline(tmp_file, num_speakers=nb_speakers)
        else:
            diarization_result = pipeline(tmp_file)

        final_result = diarize_text(asr_result, diarization_result)

        prediction = ""
        prediction_raw = list()

        for segment, speaker, sentence in final_result:
            prediction += sentence.strip()
            prediction_raw.append(
                {
                    "start": f"{segment.start:.2f}",
                    "end": f"{segment.end:.2f}",
                    "speaker": speaker,
                    "sentence": sentence.strip(),
                }
            )

    except Exception as e:
        logger.error(f"Error while running pipeline: {e}")

        return {
            "prediction": "Error while running pipeline",
            "prediction_raw": ERROR_MSG.format(e=e),
        }

    finally:
        delete_file(tmp_file)

    return {"prediction": prediction, "prediction_raw": prediction_raw}

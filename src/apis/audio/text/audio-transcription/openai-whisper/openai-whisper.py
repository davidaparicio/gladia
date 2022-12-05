from logging import getLogger
from typing import Dict

import whisper
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


error_msg = """Error while loading pipeline: {e}
    Please check your HuggingFace credentials in the environment variables HUGGINGFACE_ACCESS_TOKEN
    Also make sure that you have approved the terms of use for the segmentation and diarization models
    for the HUGGINGFACE_ACCESS_TOKEN related token
    """


@input_to_files
def predict(
    audio: str, language: str = "en", model_version: str = "tiny"
) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be transcribed.
        language (str): The language of the audio to be transcribed. (default: "en")
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
        logger.error(error_msg.format(e=e))

    audio_segment = AudioSegment.from_file(audio)

    tmp_file = get_tmp_filename()
    # unfortunately, pyannote.audio only accepts wav files
    # so we need to convert the audio to wav
    # and then delete the file
    # Bytes are said to be supported but it doesn't work
    audio_segment.export(tmp_file, format="wav")

    try:
        model = whisper.load_model(model_version)
        asr_result = model.transcribe(tmp_file)
        diarization_result = pipeline(tmp_file)
        final_result = diarize_text(asr_result, diarization_result)

        prediction_raw = list()
        for segment, speaker, sentence in final_result:
            prediction_raw.append(
                {
                    "start": f"{segment.start:.2f}",
                    "end": f"{segment.end:.2f}",
                    "speaker": speaker,
                    "sentence": sentence,
                }
            )

    except Exception as e:
        logger.error(f"Error while running pipeline: {e}")
        return {
            "prediction": "Error while running pipeline",
            "prediction_raw": error_msg.format(e=e),
        }
    finally:
        delete_file(tmp_file)

    return {"prediction": prediction_raw, "prediction_raw": prediction_raw}

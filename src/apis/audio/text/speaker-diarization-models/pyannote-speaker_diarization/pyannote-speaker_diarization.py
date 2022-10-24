from typing import Dict
from pyannote.audio import Pipeline
from gladia_api_utils.file_management import input_to_files, get_tmp_filename, delete_file
from pydub import AudioSegment
import io
from logging import getLogger

logger = getLogger(__name__)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07")

@input_to_files
def predict(audio: str) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be transcribed.

    Outputs:
        Dict[str, str]: The text of the audio splitted into segmented speakers.
    """

    audio_segment = AudioSegment.from_file(audio)

    f = get_tmp_filename()
    # unfortunately, pyannote.audio only accepts wav files
    # so we need to convert the audio to wav
    # and then delete the file
    # Bytes are said to be supported but it doesn't work
    audio_segment.export(f, format="wav")
    diarization = pipeline(f)
    delete_file(f)

    labels = diarization.labels()
    segments = list()

    for segment, _, label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "label": label
            })

    prediction = {"labels": labels, "segments": segments}

    return {"prediction": prediction, "prediction_raw": prediction}

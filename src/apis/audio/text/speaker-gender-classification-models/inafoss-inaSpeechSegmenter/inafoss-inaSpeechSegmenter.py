from typing import Dict

from inaSpeechSegmenter import Segmenter

from gladia_api_utils.file_management import input_to_files

@input_to_files
def predict(audio: bytes) -> Dict[str, str]:
    """
    Predict the text from the audio: audio -> text for a given language.

    Args:
        audio (bytes): The bytes audio to be classify the speaker gender from.

    Returns:
        Dict[str, str]: Speaker's gender.
    """
    seg = Segmenter()
    segmentation = seg(audio)

    output = []
    genders = []
    for segment in segmentation:
        if "male" in segment[0]:
            genders.append(segment[0])
        output.append({
            "type": segment[0],
            "start": segment[1],
            "end": segment[2],
            })

    # remove duplicates from gender list
    genders = list(dict.fromkeys(genders))

    return {"prediction": genders, "prediction_raw": output}

from typing import Dict, List, Union

from gladia_api_utils import SECRETS
from gladia_api_utils.file_management import input_to_files
from transformers import pipeline


@input_to_files
def predict(
    video: bytes,
    top_k: int = 5,
    model_version: str = "MCG-NJU-videomae-base-finetuned-kinetics",
) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
    """
    Predicts the class of a video using the Kinetics-600 dataset.

    Args:
        video (bytes): The video to classify.
        top_k (int): The number of classes to return.
        model_version (str): The version of the model to use.

    Returns:
        Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]: The prediction and the raw prediction.
    """

    video_classifier = pipeline(
        "video-classification",
        model=model_version,
        frame_sampling_rate=4,
        use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
    )

    prediction_raw = video_classifier(video)

    return {"prediction": prediction_raw[0]["label"], "prediction_raw": prediction_raw}

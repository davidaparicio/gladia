from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "audio",
        "name": "audio",
        "example": task_metadata["inputs"]["audio_url"]["examples"][0],
        "examples": task_metadata["inputs"]["audio_url"]["examples"],
        "placeholder": "Audio to categorize the gender from",
    },
]

output = {
    "name": "gender",
    "type": "string",
    "example": "",
}

router = APIRouter()

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="inafoss-inaSpeechSegmenter",
)

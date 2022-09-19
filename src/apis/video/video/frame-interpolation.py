from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "video",
        "name": "video",
        "example": task_metadata["inputs_example"]["video_url"]["default_example"],
        "examples": task_metadata["inputs_example"]["video_url"]["examples"],
        "placeholder": "File to the video to interpolate from",
    }
]

output = {
    "name": "interpolated_video",
    "type": "video",
    "example": "interpolated_video",
}

router = APIRouter()

TaskRouter(
    router=router,
    input=input,
    output=output,
    default_model="deep-animation-video-interpolation-in-the-wild",
)

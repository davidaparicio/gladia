from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "image",
        "name": "image",
        "example": task_metadata["inputs_example"]["image_url"]["default_example"],
        "examples": task_metadata["inputs_example"]["image_url"]["examples"],
        "placeholder": "Image to classify",
    },
    {
        "type": "integer",
        "name": "top_k",
        "default": task_metadata["inputs_example"]["top_k"]["default_example"],
        "example": task_metadata["inputs_example"]["top_k"]["default_example"],
        "examples": task_metadata["inputs_example"]["top_k"]["examples"],
        "placeholder": "Top K",
    },
]

output = {
    "name": "classified_image",
    "type": "string",
    "example": "bow tie",
}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="alexnet")

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "image",
        "name": "original_image",
        "example": task_metadata["inputs_example"]["original_image_url"][
            "default_example"
        ],
        "examples": task_metadata["inputs_example"]["original_image_url"]["examples"],
        "placeholder": "Image to replace the background from",
    },
    {
        "type": "image",
        "name": "background_image",
        "example": task_metadata["inputs_example"]["background_image_url"][
            "default_example"
        ],
        "examples": task_metadata["inputs_example"]["background_image_url"]["examples"],
        "placeholder": "Image the background will be replaced with",
    },
    {
        "type": "list",
        "name": "alignment",
        "example": task_metadata["inputs_example"]["alignment"]["default_example"],
        "examples": task_metadata["inputs_example"]["alignment"]["examples"],
        "placeholder": "original image insertion position in the background image",
    },
]

output = {"name": "replaced_image", "type": "image", "example": "a.png"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="mobilenet")

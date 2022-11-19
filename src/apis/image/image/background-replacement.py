from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": task_metadata["inputs"]["original_image_url"]["type"],
        "name": "original_image",
        "example": task_metadata["inputs"]["original_image_url"]["examples"][0],
        "examples": task_metadata["inputs"]["original_image_url"]["examples"],
        "placeholder": "Image to replace the background from",
    },
    {
        "type": task_metadata["inputs"]["background_image_url"]["type"],
        "name": "background_image",
        "example": task_metadata["inputs"]["background_image_url"]["examples"][0],
        "examples": task_metadata["inputs"]["background_image_url"]["examples"],
        "placeholder": "Image the background will be replaced with",
    },
    {
        "type": task_metadata["inputs"]["alignment"]["type"],
        "name": "alignment",
        "example": task_metadata["inputs"]["alignment"]["examples"][0],
        "examples": task_metadata["inputs"]["alignment"]["examples"],
        "placeholder": "original image insertion position in the background image",
    },
]

output = {"name": "replaced_image", "type": "image", "example": "a.png"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="mobilenet")

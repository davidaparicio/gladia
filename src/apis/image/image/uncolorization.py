from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": task_metadata["inputs"]["image_url"]["type"],
        "name": "image",
        "example": task_metadata["inputs"]["image_url"]["examples"][0],
        "examples": task_metadata["inputs"]["image_url"]["examples"],
        "placeholder": "Image to uncolorize",
    }
]

output = {"name": "uncolorized_image", "type": "image", "example": "uncolorized_image"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="v1")

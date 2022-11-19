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
        "placeholder": "Image to inpaint",
    },
    {
        "type": task_metadata["inputs"]["mask_image_url"]["type"],
        "name": "mask_image",
        "example": task_metadata["inputs"]["mask_image_url"]["examples"][0],
        "examples": task_metadata["inputs"]["mask_image_url"]["examples"],
        "placeholder": "Mask to guide inpainting",
    },
]

output = {"name": "inpainted_image", "type": "image", "example": "inpainted_image"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="zits")

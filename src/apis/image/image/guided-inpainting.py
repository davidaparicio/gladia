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
        "placeholder": "Image to inpaint",
    },
    {
        "type": "image",
        "name": "mask_image",
        "example": task_metadata["inputs_example"]["mask_image_url"]["default_example"],
        "examples": task_metadata["inputs_example"]["mask_image_url"]["examples"],
        "placeholder": "Mask to guide inpainting",
    },
    {
        "type": "string",
        "name": "prompt",
        "example": task_metadata["inputs_example"]["prompt"]["default_example"],
        "examples": task_metadata["inputs_example"]["prompt"]["examples"],
        "placeholder": "Mask to guide inpainting",
    },
]

output = {"name": "inpainted_image", "type": "image", "example": "inpainted_image"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="stable-diffusion")

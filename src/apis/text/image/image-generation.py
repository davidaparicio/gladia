from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "string",
        "name": "prompt",
        "example": task_metadata["inputs_example"]["prompt"]["default_example"],
        "examples": task_metadata["inputs_example"]["prompt"]["examples"],
        "placeholder": "Prompt to generate image from",
    },
    {
        "type": "integer",
        "name": "samples",
        "default": 1,
        "example": task_metadata["inputs_example"]["samples"]["default_example"],
        "examples": task_metadata["inputs_example"]["samples"]["examples"],
        "placeholder": "Number of predictions",
    },
    {
        "type": "integer",
        "name": "steps",
        "default": 40,
        "example": task_metadata["inputs_example"]["steps"]["default_example"],
        "examples": task_metadata["inputs_example"]["steps"]["examples"],
        "placeholder": "Number of steps",
    },
    {
        "type": "integer",
        "name": "seed",
        "default": 396916372,
        "example": task_metadata["inputs_example"]["seed"]["default_example"],
        "examples": task_metadata["inputs_example"]["seed"]["examples"],
        "placeholder": "Seed for predictions",
    },
]

output = {"name": "generatedimage", "type": "image", "example": ""}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="stable-diffusion")

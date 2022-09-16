from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "sentence",
        "example": task_metadata["inputs_example"]["sentence"]["default_example"],
        "examples": task_metadata["inputs_example"]["sentence"]["examples"],
        "placeholder": "Insert the text to find the next word from.",
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

output = {"name": "next_word", "type": "string", "example": "next word"}


TaskRouter(
    router=router, input=inputs, output=output, default_model="distilbert-base-uncased"
)

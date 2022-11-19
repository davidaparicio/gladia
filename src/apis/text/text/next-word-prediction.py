from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["sentence"]["type"],
        "name": "sentence",
        "example": task_metadata["inputs"]["sentence"]["examples"][0],
        "examples": task_metadata["inputs"]["sentence"]["examples"],
        "placeholder": "Insert the text to find the next word from.",
    },
    {
        "type": task_metadata["inputs"]["top_k"]["type"],
        "name": "top_k",
        "default": task_metadata["inputs"]["top_k"]["examples"][0],
        "example": task_metadata["inputs"]["top_k"]["examples"][0],
        "examples": task_metadata["inputs"]["top_k"]["examples"],
        "placeholder": "Top K",
    },
]

output = {"name": "next_word", "type": "string", "example": "next word"}


TaskRouter(
    router=router, input=inputs, output=output, default_model="distilbert-base-uncased"
)

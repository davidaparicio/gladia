from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["text"]["type"],
        "name": "text",
        "example": task_metadata["inputs"]["text"]["examples"][0],
        "examples": task_metadata["inputs"]["text"]["examples"],
        "placeholder": "Insert the text to summarize here",
    },
    {
        "type": task_metadata["inputs"]["source_language"]["type"],
        "name": "source_language",
        "example": task_metadata["inputs"]["source_language"]["examples"][0],
        "examples": task_metadata["inputs"]["source_language"]["examples"],
        "placeholder": "Use the ISO 3 letters representation for source language",
    },
    {
        "type": task_metadata["inputs"]["min_length"]["type"],
        "name": "min_length",
        "default": task_metadata["inputs"]["min_length"]["examples"][0],
        "example": task_metadata["inputs"]["min_length"]["examples"][0],
        "examples": task_metadata["inputs"]["min_length"]["examples"],
        "placeholder": "Minimum lenght of the summary",
    },
    {
        "type": task_metadata["inputs"]["max_length"]["type"],
        "name": "max_length",
        "default": task_metadata["inputs"]["max_length"]["examples"][0],
        "example": task_metadata["inputs"]["max_length"]["examples"][0],
        "examples": task_metadata["inputs"]["max_length"]["examples"],
        "placeholder": "Maximum lenght of the summary",
    },
]

output = {"name": "summarized_text", "type": "string", "example": "summarized_text"}

TaskRouter(router=router, input=inputs, output=output, default_model="all-MiniLM-L6-v2")

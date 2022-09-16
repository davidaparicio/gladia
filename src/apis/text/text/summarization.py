from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "text",
        "example": task_metadata["inputs_example"]["text"]["default_example"],
        "examples": task_metadata["inputs_example"]["text"]["examples"],
        "placeholder": "Insert the text to summarize here",
    },
    {
        "type": "string",
        "name": "source_language",
        "example": task_metadata["inputs_example"]["source_language"][
            "default_example"
        ],
        "examples": task_metadata["inputs_example"]["source_language"]["examples"],
        "placeholder": "Use the ISO 3 letters representation for source language",
    },
    {
        "type": "integer",
        "name": "min_length",
        "default": task_metadata["inputs_example"]["min_length"]["default_example"],
        "example": task_metadata["inputs_example"]["min_length"]["default_example"],
        "examples": task_metadata["inputs_example"]["min_length"]["examples"],
        "placeholder": "Minimum lenght of the summary",
    },
    {
        "type": "integer",
        "name": "max_length",
        "default": task_metadata["inputs_example"]["max_length"]["default_example"],
        "example": task_metadata["inputs_example"]["max_length"]["default_example"],
        "examples": task_metadata["inputs_example"]["max_length"]["examples"],
        "placeholder": "Maximum lenght of the summary",
    },
]

output = {"name": "summarized_text", "type": "string", "example": "summarized_text"}

TaskRouter(router=router, input=inputs, output=output, default_model="all-MiniLM-L6-v2")

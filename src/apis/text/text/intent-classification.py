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
        "placeholder": "Insert the text to extract intent and slot from",
    }
]

output = {"name": "analyzed_text", "type": "array", "example": "analyzed_text"}


# TaskRouter(router=router, input=inputs, output=output, default_model="jointBERT-bert-atis")

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
        "placeholder": "Insert the text to transliterate here",
    },
    {
        "type": task_metadata["inputs"]["language"]["type"],
        "name": "language",
        "example": task_metadata["inputs"]["language"]["examples"][0],
        "examples": task_metadata["inputs"]["language"]["examples"],
        "placeholder": "Insert the language code here",
    },
]

output = {
    "name": "transliterated_text",
    "type": "string",
    "example": "transliterated_text",
}

TaskRouter(router=router, input=inputs, output=output, default_model="transliterate")

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "context",
        "example": task_metadata["inputs"]["context"]["examples"][0],
        "examples": task_metadata["inputs"]["context"]["examples"],
        "placeholder": "Insert the text to extract answer from",
    },
    {
        "type": "string",
        "name": "question",
        "example": task_metadata["inputs"]["question"]["examples"][0],
        "examples": task_metadata["inputs"]["question"]["examples"],
        "placeholder": "Insert the question to be answered",
    },
    {
        "type": "integer",
        "name": "top_k",
        "default": task_metadata["inputs"]["top_k"]["examples"][0],
        "example": task_metadata["inputs"]["top_k"]["examples"][0],
        "examples": task_metadata["inputs"]["top_k"]["examples"],
        "placeholder": "Top K",
    },
]

output = {"name": "answer", "type": "string", "example": "answer"}


TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="distilbert-base-cased-distilled-squad",
)

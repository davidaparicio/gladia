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
        "type": "integer",
        "name": "top_k",
        "default": task_metadata["inputs_example"]["top_k"]["default_example"],
        "example": task_metadata["inputs_example"]["top_k"]["default_example"],
        "examples": task_metadata["inputs_example"]["top_k"]["examples"],
        "placeholder": "Top K",
    },
]

output = {"name": "keywords", "type": "string", "example": "crown"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="keybert-paraphrase-MiniLM-L6-v2",
)

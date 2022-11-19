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
        "type": task_metadata["inputs"]["top_k"]["type"],
        "name": "top_k",
        "default": task_metadata["inputs"]["top_k"]["examples"][0],
        "example": task_metadata["inputs"]["top_k"]["examples"][0],
        "examples": task_metadata["inputs"]["top_k"]["examples"],
        "placeholder": "Top K",
    },
]

output = {"name": "keywords", "type": "string", "example": "crown"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="keybert-paraphrase-multilingual-MiniLM-L12-v2",
)

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "text",
        "example": task_metadata["inputs"]["text"]["examples"][0],
        "examples": task_metadata["inputs"]["text"]["examples"],
        "placeholder": "Insert the text to anlayse sentiment from",
    }
]

output = {"name": "sentiment", "type": "string", "example": "POSITIVE"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="distilbert-base-uncased-finetuned-sst-2-english",
)

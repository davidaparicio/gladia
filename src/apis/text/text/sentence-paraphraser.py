from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "context",
        "example": task_metadata["inputs_example"]["context"]["default_example"],
        "examples": task_metadata["inputs_example"]["context"]["examples"],
        "placeholder": "Insert the text to paraphrase here",
    }
]

output = {"name": "paraphrased_text", "type": "string", "example": "paraphrased_text"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="ramsrigouthamg-t5-large-paraphraser-diverse-high-quality",
)

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
        "placeholder": "Insert the text to anonymize",
    },
    {
        "type": "string",
        "name": "language",
        "example": task_metadata["inputs_example"]["language"]["default_example"],
        "examples": task_metadata["inputs_example"]["language"]["examples"],
        "placeholder": "Insert the language of the text to anonymize",
    },
    {
        "type": "string",
        "name": "entities",
        "example": task_metadata["inputs_example"]["entities"]["default_example"],
        "examples": task_metadata["inputs_example"]["entities"]["examples"],
        "placeholder": "Entities to anonymize (default: None=all)",
    }
]

output = {"name": "anonymized_text", "type": "string", "example": "anonymized_text"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="ms-presidio",
)

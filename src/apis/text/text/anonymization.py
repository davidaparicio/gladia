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
        "placeholder": "Insert the text to anonymize",
    },
    {
        "type": "string",
        "name": "language",
        "example": task_metadata["inputs"]["language"]["examples"][0],
        "examples": task_metadata["inputs"]["language"]["examples"],
        "placeholder": "Insert the language of the text to anonymize",
    },
    {
        "type": "string",
        "name": "entities",
        "example": task_metadata["inputs"]["entities"]["examples"][0],
        "examples": task_metadata["inputs"]["entities"]["examples"],
        "placeholder": "Entities to anonymize (default: None=all)",
    },
]

output = {"name": "anonymized_text", "type": "string", "example": "anonymized_text"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="ms-presidio",
)

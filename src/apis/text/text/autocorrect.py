from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "sentence",
        "example": task_metadata["inputs"]["sentence"]["examples"][0],
        "examples": task_metadata["inputs"]["sentence"]["examples"],
        "placeholder": "Insert the text to correct",
    }
]

output = {"name": "corrected_text", "type": "string", "example": "corrected_text"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="flexudy-t5-base-multi-sentence-doctor",
)

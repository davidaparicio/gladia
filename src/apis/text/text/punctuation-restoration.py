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
        "placeholder": "Insert the text to restore punctation from",
    }
]

output = {"name": "restored_sentence", "type": "string", "example": "restored_sentence"}

TaskRouter(
    router=router, input=inputs, output=output, default_model="kredor-punctuate-all"
)

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["input_string"]["type"],
        "name": "input_string",
        "example": task_metadata["inputs"]["input_string"]["examples"][0],
        "examples": task_metadata["inputs"]["input_string"]["examples"],
        "placeholder": "Insert the text to analyze here",
    }
]

output = {"name": "analyzed_text", "type": "string", "example": "analyzed_text"}


TaskRouter(router=router, input=inputs, output=output, default_model="LAL-Parser")

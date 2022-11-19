from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["code_snippet"]["type"],
        "name": "code_snippet",
        "example": task_metadata["inputs"]["code_snippet"]["examples"][0],
        "examples": task_metadata["inputs"]["code_snippet"]["examples"],
        "placeholder": "Insert the code to generate from",
    }
]

output = {"name": "generated_code", "type": "string", "example": "generated_code"}

TaskRouter(router=router, input=inputs, output=output, default_model="sentdex-GPyT")

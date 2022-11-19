from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [{
    "name": input_name,
    "type": task_metadata["inputs"][input_name]["type"],
    "default": task_metadata["inputs"][input_name].get("default", ...),
    "example": task_metadata["inputs"][input_name]["examples"][0],
    "examples": task_metadata["inputs"][input_name]["examples"],
    "placeholder": task_metadata["inputs"][input_name]["placeholder"],
} for input_name in task_metadata["inputs"]]

output = {"name": "analyzed_text", "type": "string", "example": "analyzed_text"}


TaskRouter(router=router, input=inputs, output=output, default_model="LAL-Parser")

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "date",
        "example": task_metadata["inputs_example"]["date"]["default_example"],
        "examples": task_metadata["inputs_example"]["date"]["examples"],
        "placeholder": "Insert the date to format",
    }
]

output = {"name": "formatted_date", "type": "string", "example": "formatted_date"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="dataprep-formatter",
)
from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "address",
        "example": task_metadata["inputs_example"]["address"]["default_example"],
        "examples": task_metadata["inputs_example"]["address"]["examples"],
        "placeholder": "Insert the address to format",
    }
]

output = {"name": "formated_address", "type": "string", "example": "formated_address"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="geopy-formatter",
)

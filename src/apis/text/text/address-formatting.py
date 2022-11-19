from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["address"]["type"],
        "name": "address",
        "example": task_metadata["inputs"]["address"]["examples"][0],
        "examples": task_metadata["inputs"]["address"]["examples"],
        "placeholder": "Insert the address to format",
    }
]

output = {"name": "formatted_address", "type": "string", "example": "formatted_address"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="geopy-formatter",
)

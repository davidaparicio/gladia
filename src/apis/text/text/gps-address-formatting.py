from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "float",
        "name": "latitude",
        "example": task_metadata["inputs_example"]["latitude"]["default_example"],
        "examples": task_metadata["inputs_example"]["latitude"]["examples"],
        "placeholder": "Insert the latitude of the address to fetch",
    },
    {
        "type": "float",
        "name": "longitude",
        "example": task_metadata["inputs_example"]["longitude"]["default_example"],
        "examples": task_metadata["inputs_example"]["longitude"]["examples"],
        "placeholder": "Insert the longitude of the address to fetch",
    },
]

output = {"name": "formated_address", "type": "string", "example": "formated_address"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="geopy-formatter",
)

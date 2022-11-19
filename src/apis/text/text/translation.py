from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "name": input_name,
        "type": task_metadata["inputs"][input_name]["type"],
        "default": task_metadata["inputs"][input_name].get("default", ...),
        "example": task_metadata["inputs"][input_name]["examples"][0],
        "examples": task_metadata["inputs"][input_name]["examples"],
        "placeholder": task_metadata["inputs"][input_name]["placeholder"],
    }
    for input_name in task_metadata["inputs"]
]

output = {
    "name": "translated_text",
    "type": "str",
    "example": '{"prediction": "Le texte à traduire",  "prediction_raw": "Le texte à traduire"}',
}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="facebook-nllb-200-distilled-600M",
)

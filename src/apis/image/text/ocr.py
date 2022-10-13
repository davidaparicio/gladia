from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": "image",
        "name": "image",
        "example": task_metadata["inputs_example"]["image_url"]["default_example"],
        "examples": task_metadata["inputs_example"]["image_url"]["examples"],
        "placeholder": "Image to extract text from",
    },
    {
        "type": "string",
        "name": "source_language",
        "default": task_metadata["inputs_example"]["source_language"][
            "default_example"
        ],
        "example": task_metadata["inputs_example"]["source_language"][
            "default_example"
        ],
        "examples": task_metadata["inputs_example"]["source_language"]["examples"],
        "placeholder": "ISO 639-2 Source language (3 letters)",
    },
]

output = {"name": "extracted_text", "type": "string", "example": "extracted_text"}

router = APIRouter()

TaskRouter(
    router=router, input=inputs, output=output, default_model="textract-extractor"
)

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "input_string",
        "example": task_metadata["inputs"]["input_string"]["examples"][0],
        "examples": task_metadata["inputs"]["input_string"]["examples"],
        "placeholder": "Insert the text to translate here",
    },
    {
        "type": "string",
        "name": "source_language",
        "example": task_metadata["inputs"]["source_language"]["examples"][0],
        "examples": task_metadata["inputs"]["source_language"]["examples"],
        "placeholder": "Use the ISO 3 letters (ISO 639-3) representation for source language",
    },
    {
        "type": "string",
        "name": "target_language",
        "example": task_metadata["inputs"]["target_language"]["examples"][0],
        "examples": task_metadata["inputs"]["target_language"]["examples"],
        "placeholder": "Use the ISO 3 letters (ISO 639-3) representation for target language",
    },
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

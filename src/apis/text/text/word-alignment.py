from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "input_string_language_1",
        "example": task_metadata["inputs"]["input_string_language_1"]["examples"][0],
        "examples": task_metadata["inputs"]["input_string_language_1"][
            "examples"
        ],
        "placeholder": "Insert the Sentence from first language",
    },
    {
        "type": "string",
        "name": "input_string_language_2",
        "example": task_metadata["inputs"]["input_string_language_2"]["examples"][0],
        "examples": task_metadata["inputs"]["input_string_language_2"][
            "examples"
        ],
        "placeholder": "Insert the Sentence from second language",
    },
]

output = {
    "name": "word_aligment",
    "type": "array",
    "example": '[{"source": "Sentence","target": "来自"}]',
}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="bert-base-multilingual-cased",
)

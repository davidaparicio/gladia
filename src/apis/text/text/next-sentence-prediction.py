from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["sentence_1"]["type"],
        "name": "sentence_1",
        "example": task_metadata["inputs"]["sentence_1"]["examples"][0],
        "examples": task_metadata["inputs"]["sentence_1"]["examples"],
        "placeholder": "Insert the first sentence",
    },
    {
        "type": task_metadata["inputs"]["sentence_2"]["type"],
        "name": "sentence_2",
        "example": task_metadata["inputs"]["sentence_2"]["examples"][0],
        "examples": task_metadata["inputs"]["sentence_2"]["examples"],
        "placeholder": "Insert the second sentence to estimate the probability from",
    },
]

output = {
    "name": "next_sentence_probability",
    "type": "number",
    "example": "0.999984622001648",
}

TaskRouter(
    router=router, input=inputs, output=output, default_model="bert-base-uncased"
)

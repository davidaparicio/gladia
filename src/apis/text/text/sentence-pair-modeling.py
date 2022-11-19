from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["sentence"]["type"],
        "name": "sentence",
        "example": task_metadata["inputs"]["sentence"]["examples"][0],
        "examples": task_metadata["inputs"]["sentence"]["examples"],
        "placeholder": "Insert the sentence to perform the Pairwise Sentence Scoring Tasks",
    }
]

output = {"name": "analyzed_sentence", "type": "string", "example": "analyzed_sentence"}

TaskRouter(
    router=router, input=inputs, output=output, default_model="UKPLab-all-MiniLM-L6-v2"
)

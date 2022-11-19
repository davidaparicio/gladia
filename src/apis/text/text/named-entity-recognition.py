from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

router = APIRouter()

inputs = [
    {
        "type": task_metadata["inputs"]["text"]["type"],
        "name": "text",
        "example": task_metadata["inputs"]["text"]["examples"][0],
        "examples": task_metadata["inputs"]["text"]["examples"],
        "placeholder": "Insert the text to anlayse sentiment from",
    }
]

output = {
    "name": "recognized_entities",
    "type": "array",
    "example": '{"prediction":[{"entity_group": "ORG", "score": 0.5587025284767151, "word": "Gladia", "start": 26, "end": 32}]',
}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="dbmdz-bert-large-cased-finetuned-conll03-english",
)

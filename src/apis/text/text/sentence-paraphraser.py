from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter

router = APIRouter()

inputs = [
    {
        "type": "string",
        "name": "context",
        "example": "Once, a group of frogs was roaming around the forest in search of water.",
        "placeholder": "Insert the text to paraphrase here",
    }
]

output = {"name": "paraphrased_text", "type": "string", "example": "paraphrased_text"}

TaskRouter(
    router=router,
    input=inputs,
    output=output,
    default_model="ramsrigouthamg-t5-large-paraphraser-diverse-high-quality",
)

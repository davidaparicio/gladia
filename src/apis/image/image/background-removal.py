from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter

inputs = [
    {
        "type": "image",
        "name": "image",
        "default": "http://files.gladia.io/test/test.png",
        "placeholder": "Image to remove the background from",
    }
]

output = {"name": "cleaned_image", "type": "image", "example": "a.png"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="xception")

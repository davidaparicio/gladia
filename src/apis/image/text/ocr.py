from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter

inputs = [
    {
        "type": "image",
        "name": "image",
        "default": "http://files.gladia.io/test/test.png",
        "placeholder": "Image to extract text from",
    },
    {
        "type": "text",
        "name": "source_language",
        "default": "en",
    },
]

output = {"name": "extracted_text", "type": "list", "example": "extracted_text"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="easy-ocr")

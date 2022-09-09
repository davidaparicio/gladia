from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter

inputs = [
    {
        "type": "image",
        "name": "image",
        "example": "http://files.gladia.io/examples/image/image/colorization/rsz_paris.jpg",
        "examples": [
            "http://files.gladia.io/examples/image/image/colorization/rsz_paris.gif",
            "http://files.gladia.io/examples/image/image/colorization/rsz_paris.jpg",
            "http://files.gladia.io/examples/image/image/colorization/rsz_paris.png",
        ],
        "placeholder": "Image to colorize",
    }
]

output = {"name": "colorized_image", "type": "image", "example": "image"}

router = APIRouter()

TaskRouter(
    router=router, input=inputs, output=output, default_model="mlhubber_colorize"
)

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter

inputs = [
    {
        "type": "image",
        "name": "original_image",
        "example": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        "placeholder": "Image to inpaint",
    },
    {
        "type": "image",
        "name": "mask_image",
        "example": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png",
        "placeholder": "Mask to guide inpainting",
    },
]

output = {"name": "inpainted_image", "type": "image", "example": "inpainted_image"}

router = APIRouter()

TaskRouter(router=router, input=inputs, output=output, default_model="zits")

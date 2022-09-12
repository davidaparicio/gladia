from io import BytesIO

import PIL
import requests
import torch
from diffusers import StableDiffusionInpaintPipeline
from gladia_api_utils import SECRETS
from gladia_api_utils.io import _open
from torch import autocast


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content))


def predict(original_image: bytes, mask_image: bytes, prompt: str = ""):
    original_image = _open(original_image).convert("RGB").resize((512, 512))
    mask_image = _open(mask_image).convert("RGB").resize((512, 512))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
    )

    pipe = pipe.to(device)

    with autocast("cuda"):
        images = pipe(
            prompt=prompt,
            init_image=original_image,
            mask_image=mask_image,
            strength=0.75,
        ).images

    return images[0]

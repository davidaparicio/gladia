import requests
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import torch_device
from gladia_api_utils import SECRETS
from gladia_api_utils.io import _open
from numpy import size
from torch import autocast
from torchvision import transforms


def predict(
    original_image: bytes,
    mask_image: bytes,
    example_image: bytes,
    seed: int = 424242,
    steps: int = 75,
    guidance_scale: int = 15,
):

    original_image = _open(original_image)
    width, height = original_image.size
    original_image = original_image.convert("RGB").resize((512, 512))
    example_image = _open(example_image).convert("RGB").resize((512, 512))
    mask_image = _open(mask_image).convert("RGB").resize((512, 512))

    device = "cuda"
    model_id = "Fantasy-Studio/Paint-by-Example"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
    )

    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(seed) if seed != 0 else None

    with autocast(device):
        images = pipe(
            image=original_image,
            example_image=example_image,
            mask_image=mask_image,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images

    return images[0].resize((width, height))

import torch
from diffusers import StableDiffusionInpaintPipeline
from gladia_api_utils import SECRETS
from gladia_api_utils.io import _open
from numpy import size
from torch import autocast


def predict(original_image: bytes, mask_image: bytes, prompt: str = ""):
    original_image = _open(original_image)
    width, height = original_image.size
    original_image = original_image.convert("RGB").resize((512, 512))
    mask_image = _open(mask_image).convert("RGB").resize((512, 512))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id_or_path = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
    )

    pipe = pipe.to(device)

    with autocast("cuda"):
        images = pipe(prompt=prompt, image=original_image, mask_image=mask_image).images

    return images[0].resize((width, height))

from logging import getLogger
from typing import List, Union

import torch
from diffusers import StableDiffusionPipeline
from gladia_api_utils import SECRETS
from gladia_api_utils.image_management import convert_image_to_base64
from PIL import Image
from torch import autocast

logger = getLogger(__name__)

MODEL_VERSIONS = {
    "compvis-sd-14": "CompVis/stable-diffusion-v1-4",
    "runwayml-sd-15": "runwayml/stable-diffusion-v1-5",
    "stabilityai-sd-20": "stabilityai/stable-diffusion-2",
}


def predict(
    prompt: str = "A high tech solarpunk utopia in the Amazon rainforest",
    samples: int = 1,
    steps: int = 40,
    scale: float = 7.5,
    seed: int = 396916372,
    model_version: str = "stabilityai-sd-20",
) -> Union[Image.Image, List[str]]:
    """
    Generate an image using the the stable diffusion model.
    NSFW filter not implemented yet.
    /!\ If samples > 1, the output will be a list of base 64 images instead of a single Pil Image.

    Args:
        prompt (str): The prompt to use for the generation
        samples (int): The number of samples to generate. (default: 1)
        steps (int): The number of steps to use for the generation (higher is better)
        scale (float): The scale to use for the generation (recommended between 0.0 and 15.0)
        seed (int): The seed to use for the generation (default: 396916372)
        model_version (str): The model version to use. (default: stabilityai-sd-20)

    Returns:
        Union[Image.Image, List[Image.Image]]: The generated image if samples=1, else a list of generated images in a base64 format
    """

    model_id = MODEL_VERSIONS[model_version]
    device = "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=SECRETS["HUGGINGFACE_ACCESS_TOKEN"],
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    with autocast("cuda"):
        images_list = pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )

    torch.cuda.empty_cache()
    if samples == 1:
        return images_list["images"][0]
    else:
        output = list()
        for image in images_list["images"]:
            output.append(convert_image_to_base64(image))
        return output

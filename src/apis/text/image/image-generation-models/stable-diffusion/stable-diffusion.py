import torch
from diffusers import StableDiffusionPipeline
from gladia_api_utils import SECRETS
from PIL import Image
from torch import autocast
from gladia_api_utils.image_management import Image_to_base64
from logging import getLogger

logger = getLogger(__name__)

def predict(
    prompt="A high tech solarpunk utopia in the Amazon rainforest",
    samples=1,
    steps=40,
    scale=7.5,
    seed=396916372,
) -> Image:
    """
    Generate an image using the the stable diffusion model.
    NSFW filter not implemented yet.

    Args:
        prompt (str): The prompt to use for the generation
        samples (int): The number of samples to generate. >1 Not supported so far (default: 1)
        steps (int): The number of steps to use for the generation (higher is better)
        scale (float): The scale to use for the generation (recommended between 0.0 and 15.0)
        seed (int): The seed to use for the generation (default: 396916372)

    Returns:
        Image: The generated image
    """

    model_id = "CompVis/stable-diffusion-v1-4"
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

    # TODO implement NSFW filter
    # {'sample': [<PIL.Image.Image image mode=RGB size=512x512 at 0x7F546A97A070>], 'nsfw_content_detected': [False]}

    if samples == 1:
        return images_list["sample"][0]
    else:
        output = list()
        for image in images_list["sample"]:
            output.append(Image_to_base64(image))
        return output
        
                
    

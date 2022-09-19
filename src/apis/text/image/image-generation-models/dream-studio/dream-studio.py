import io
from logging import getLogger
from typing import List, Union

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from gladia_api_utils import SECRETS
from gladia_api_utils.image_management import convert_image_to_base64
from PIL import Image
from stability_sdk import client

logger = getLogger(__name__)


def predict(
    prompt="A high tech solarpunk utopia in the Amazon rainforest",
    samples=1,
    steps=40,
    scale=7.0,
    seed=396916372,
) -> Union[Image.Image, List[str]]:
    """
    Generate an image using the generation service from the dream studio project.
    Returns a SFW PIL image.
    if NSFW will return a NSFW Warning PIL image.
    to be used this function you need to have a valid STABILITY_API_KEY set in the environment variables.
    get the STABILITY_API_KEY at https://beta.dreamstudio.ai/dream/membership

    /!\ If the samples is greater than 1, the function will return a list of base64 images else a single PIL Image that
    will then be casted as a PNG binary image.

    Args:
        prompt (str): The prompt to use for the generation service
        samples (int): The number of samples to generate from the generation service. (default: 1)
        steps (int): The number of steps to use for the generation service (higher is better)
        scale (float): The scale to use for the generation service (recommended between 0.0 and 15.0)
        seed (int): The seed to use for the generation service (default: 396916372)

    Returns:
        Union[Image.Image, List[str]]: A PIL Image if samples=1 or a list of base64 images if samples > 1
    """

    stability_api = client.StabilityInference(
        key=SECRETS["STABILITY_KEY"],
        verbose=True,
    )

    output_base64_list = list()
    for resp in stability_api.generate(
        prompt=prompt, samples=samples, steps=steps, cfg_scale=scale
    ):
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                logger.warning(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again."
                )
                img = Image.open("unsafe.png")

            elif artifact.type == generation.ARTIFACT_IMAGE:
                bytes_img = io.BytesIO(artifact.binary)
                img = Image.open(bytes_img)

            else:
                continue

            output_base64_list.append(convert_image_to_base64(img))

    if samples == 1:
        return img
    else:
        return output_base64_list

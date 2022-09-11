from enum import Enum

from pydantic import BaseModel


class HDStrategy(str, Enum):
    """
    The strategy to use when the image is too big to be processed by the model.
    """

    ORIGINAL = "Original"
    RESIZE = "Resize"
    CROP = "Crop"


class LDMSampler(str, Enum):
    """
    The LDM sampler to use.
    """

    ddim = "ddim"
    plms = "plms"


class Config(BaseModel):
    """
    The configuration of the model.
    """

    ldm_steps: int
    ldm_sampler: str = LDMSampler.plms
    zits_wireframe: bool = True
    hd_strategy: str
    hd_strategy_crop_margin: int
    hd_strategy_crop_trigger_size: int
    hd_strategy_resize_limit: int

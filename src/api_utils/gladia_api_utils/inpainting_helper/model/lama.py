import os
from logging import getLogger

import cv2
import numpy as np
import torch

logger = getLogger(__name__)

from ..helper import download_model, get_cache_path_by_url, norm_img, pad_img_to_modulo
from ..schema import Config
from .base import InpaintModel

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


class LaMa(InpaintModel):
    """
    Class to handle the inpainting of the image using LaMa model

    Args:
        device (torch.device): device to run the model on

    Inherit from InpaintModel
    """

    pad_mod = 8

    def __init__(self, device) -> None:
        """
        Initialize the model with the given device

        Args:
            device (torch.device): device to run the model on

        Returns:
            None
        """
        super().__init__(device)
        self.device = device

    def init_model(self, device: torch.device) -> None:
        """
        Initialize the model with the given device

        Args:
            device (torch.device): device to run the model on

        Returns:
            None
        """

        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        logger.info(f"Load LaMa model from: {model_path}")

        model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(device)
        model.eval()
        self.model = model
        self.model_path = model_path

    @staticmethod
    def is_downloaded() -> bool:
        """
        Check if the model is downloaded

        Returns:
            bool: True if the model is downloaded
        """
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: Config
    ) -> np.ndarray:
        """
        Forward pass of the model to get the inpainted image
        Input image and output image have same size

        Args:
            image (np.ndarray): image to be inpainted with shape (H, W, C) RGB
            mask (np.ndarray): mask of the image with shape (H, W)
            config (Config): config of the model (see schema.py) (unused)

        Returns:
            np.ndarray: inpainted image BGR
        """

        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

        return cur_res

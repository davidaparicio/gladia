import abc
from logging import getLogger
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from ..helper import boxes_from_mask, pad_img_to_modulo, resize_max_size
from ..schema import Config, HDStrategy

logger = getLogger(__name__)


class InpaintModel:
    """
    Inpainting model interface

    Args:
        device (torch.device): torch device to use (cpu or cuda)

    Returns:
        InpaintModel: The Inpainting model
    """

    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False

    def __init__(self, device: torch.device) -> None:
        """
        Args:
            device (torch.device): torch device to use (cpu or cuda)

        Returns:
            InpaintModel: The Inpainting model
        """

        self.device = device
        self.init_model(device)

    @abc.abstractmethod
    def init_model(self, device: torch.device) -> None:
        """
        Initialize the model

        Args:
            device (torch.device): torch device to use (cpu or cuda)

        Returns:
            None
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        """
        Check if the model is downloaded
        Returns:
            bool: True if the model is downloaded, False otherwise

        Raises:
            NotImplementedError: If the model is not implemented
        """
        ...

    @abc.abstractmethod
    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: Config
    ) -> torch.Tensor:
        """
        Input images and output images have same size

        Args:
            image (np.ndarray): Original Image representation in numpy array with [H, W, C] RGB format with values in [0, 255] range.
            mask (np.ndarray): Mask Image representation in numpy array with [H, W, 1] format with values in [0, 255] range.
            config (Config): Config object used for the model containing all the parameters (see schema.py)

        Returns:
            torch.Tensor: Inpainted image with [H, W, C] BGR format with values in [0, 255] range.
        """
        ...

    def _pad_forward(
        self, image: np.ndarray, mask: np.ndarray, config: Config
    ) -> torch.Tensor:
        """
        Pad image and mask to be divisible by self.pad_mod and run forward

        Args:
            image (np.ndarray): Original Image representation in numpy array with [H, W, C] RGB format with values in [0, 255] range.
            mask (np.ndarray): Mask Image representation in numpy array with [H, W, 1] format with values in [0, 255] range.
            config (Config): Config object used for the model containing all the parameters (see schema.py)

        Returns:
            torch.Tensor: Inpainted image with [H, W, C] BGR format with values in [0, 255] range.
        """

        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        logger.debug(f"final forward pad size: {pad_image.shape}")

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        original_pixel_indices = mask < 127
        result[original_pixel_indices] = image[:, :, ::-1][original_pixel_indices]

        return result

    @torch.no_grad()
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, config: Config
    ) -> torch.Tensor:
        """
        Make the instance behave live a function to inpaint the Input images using the mask and the associated config.

        Args:
            image (np.ndarray): Original Image representation in numpy array with [H, W, C] RGB format with values in [0, 255] range.
            mask (np.ndarray): Mask Image representation in numpy array with [H, W, 1] format with values in [0, 255] range.
            config (Config): Config object used for the model containing all the parameters (see schema.py)

        Returns:
            torch.Tensor: Inpainted image with [H, W, C] BGR format with values in [0, 255] range.
        """

        inpaint_result = None
        logger.debug(f"hd_strategy: {config.hd_strategy}")

        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.debug(f"Run crop strategy")
                boxes = boxes_from_mask(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                inpaint_result = image[:, :, ::-1]
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                logger.debug(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )
                inpaint_result = self._pad_forward(
                    downsize_image, downsize_mask, config
                )

                # only paste masked area result
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                original_pixel_indices = mask < 127
                inpaint_result[original_pixel_indices] = image[:, :, ::-1][
                    original_pixel_indices
                ]

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    def _crop_box(
        self, image: np.ndarray, mask: np.ndarray, box: Tuple, config: Config
    ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Crop image and mask to the box and run forward

        Args:
            image (np.ndarray): Original Image representation in numpy array with [H, W, C] RGB format with values in [0, 255] range.
            mask (np.ndarray): Mask Image representation in numpy array with [H, W, 1] format with values in [0, 255] range.
            box (Tuple): (left,top,right,bottom) box coordinates to crop the image and mask to
            config (Config): Config object used for the model containing all the parameters (see schema.py)

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple]: Cropped image, cropped mask and box coordinates
        """

        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        logger.debug(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _run_box(
        self, image: np.ndarray, mask: np.ndarray, box: Tuple, config: Config
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Crop image and mask to the box and run forward

        Args:
            image (np.ndarray): Original Image representation in numpy array with [H, W, C] RGB format with values in [0, 255] range.
            mask (np.ndarray): Mask Image representation in numpy array with [H, W, 1] format with values in [0, 255] range.
            box (Tuple): (left,top,right,bottom) box coordinates to crop the image and mask to
            config (Config): Config object used for the model containing all the parameters (see schema.py)

        Returns:
            Tuple[np.ndarray, Tuple]: Cropped image and box coordinates
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]

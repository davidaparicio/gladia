import os
import time
from logging import getLogger
from typing import List, Tuple, Union

import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F

from ..helper import get_cache_path_by_url, load_jit_model
from ..schema import Config
from .base import InpaintModel

logger = getLogger(__name__)

ZITS_INPAINT_MODEL_URL = os.environ.get(
    "ZITS_INPAINT_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-inpaint-0717.pt",
)

ZITS_EDGE_LINE_MODEL_URL = os.environ.get(
    "ZITS_EDGE_LINE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-edge-line-0717.pt",
)

ZITS_STRUCTURE_UPSAMPLE_MODEL_URL = os.environ.get(
    "ZITS_STRUCTURE_UPSAMPLE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-structure-upsample-0717.pt",
)

ZITS_WIRE_FRAME_MODEL_URL = os.environ.get(
    "ZITS_WIRE_FRAME_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-wireframe-0717.pt",
)


def resize(
    img: np.ndarray, height: int, width: int, center_crop: bool = False
) -> np.ndarray:
    """
    Resize image to a given size

    Args:
        img (np.ndarray): Image to resize
        height (int): Height of the resized image
        width (int): Width of the resized image
        center_crop (bool): If True, center crop the image to the given size. If False, resize the image to the given size.

    Returns:
        np.ndarray: Resized image
    """
    imgh, imgw = img.shape[0:2]

    if center_crop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j : j + side, i : i + side, ...]

    if imgh > height and imgw > width:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_LINEAR
    img = cv2.resize(img, (height, width), interpolation=inter)

    return img


def to_tensor(img: np.ndarray, scale: bool = True, norm: bool = False) -> torch.Tensor:
    """
    Convert image to torch tensor

    Args:
        img (np.ndarray): Image to convert
        scale (bool): If True, scale the image to [0, 1] range. If False, keep the image in [0, 255] range.

    Returns:
        torch.Tensor: Converted image to Tensor
    """

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    c = img.shape[-1]

    if scale:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
    else:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

    if norm:
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(c, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(c, 1, 1)
        img_t = (img_t - mean) / std

    return img_t


def load_masked_position_encoding(mask: np.ndarray) -> torch.Tensor:
    """
    Load masked position encoding

    Args:
        mask (np.ndarray): Mask to load

    Returns:
        torch.Tensor: Masked position encoding
    """

    ones_filter = np.ones((3, 3), dtype=np.float32)
    d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    str_size = 256
    pos_num = 128

    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1.0 - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0

    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * pos_num).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, pos_num - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct


def load_image(
    img: np.ndarray, mask: np.ndarray, device: torch.device, sigma256: float = 3.0
):
    """
    Load image and mask to device

    Args:
        img (np.ndarray): Image to load in [H, W, C] RGB format.
        mask (np.ndarray): Mask to load in [H, W] 255 format.
        device (torch.device): Device to load the image to
        sigma256 (float): Sigma for gaussian blur

    Returns:
        torch.Tensor: Loaded image
    """

    imgh, imgw = img.shape[0:2]
    img_256 = resize(img, 256, 256)

    mask = (mask > 127).astype(np.uint8) * 255
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask_256[mask_256 > 0] = 255

    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512[mask_512 > 0] = 255

    gray_256 = skimage.color.rgb2gray(img_256)
    edge_256 = skimage.feature.canny(gray_256, sigma=sigma256, mask=None).astype(float)

    img_512 = resize(img, 512, 512)

    rel_pos, abs_pos, direct = load_masked_position_encoding(mask)

    batch = dict()
    batch["images"] = to_tensor(img.copy()).unsqueeze(0).to(device)
    batch["img_256"] = to_tensor(img_256, norm=True).unsqueeze(0).to(device)
    batch["masks"] = to_tensor(mask).unsqueeze(0).to(device)
    batch["mask_256"] = to_tensor(mask_256).unsqueeze(0).to(device)
    batch["mask_512"] = to_tensor(mask_512).unsqueeze(0).to(device)
    batch["edge_256"] = to_tensor(edge_256, scale=False).unsqueeze(0).to(device)
    batch["img_512"] = to_tensor(img_512).unsqueeze(0).to(device)
    batch["rel_pos"] = torch.LongTensor(rel_pos).unsqueeze(0).to(device)
    batch["abs_pos"] = torch.LongTensor(abs_pos).unsqueeze(0).to(device)
    batch["direct"] = torch.LongTensor(direct).unsqueeze(0).to(device)
    batch["h"] = imgh
    batch["w"] = imgw

    return batch


def to_device(data: dict, device: torch.device) -> Union[dict, list]:
    """
    Move data to torch device

    Args:
        data (dict): Data to move
        device (torch.device): Device to move to

    Returns:
        Union[dict, list]: Moved data
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


class ZITS(InpaintModel):
    """
    ZITS model class for inpainting

    Args:
        device (torch.device): Device to run the model on

    """

    min_size = 256
    pad_mod = 32
    pad_to_square = True

    def __init__(self, device: torch.device) -> None:
        """
        Init ZITS model

        Args:
            device (torch.device): Device to run the model on

        Returns:
            None
        """
        super().__init__(device)
        self.device = device
        self.sample_edge_line_iterations = 1

    def init_model(self, device: torch.device) -> None:
        """
        Init the model

        Args:
            device (torch.device): Device to run the model on

        Returns:
            None
        """

        self.wireframe = load_jit_model(ZITS_WIRE_FRAME_MODEL_URL, device)
        self.edge_line = load_jit_model(ZITS_EDGE_LINE_MODEL_URL, device)
        self.structure_upsample = load_jit_model(
            ZITS_STRUCTURE_UPSAMPLE_MODEL_URL, device
        )
        self.inpaint = load_jit_model(ZITS_INPAINT_MODEL_URL, device)

    @staticmethod
    def is_downloaded() -> bool:
        """
        Check if the model is downloaded

        Returns:
            bool: True if downloaded
        """

        model_paths = [
            get_cache_path_by_url(ZITS_WIRE_FRAME_MODEL_URL),
            get_cache_path_by_url(ZITS_EDGE_LINE_MODEL_URL),
            get_cache_path_by_url(ZITS_STRUCTURE_UPSAMPLE_MODEL_URL),
            get_cache_path_by_url(ZITS_INPAINT_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def wireframe_edge_and_line(self, items: dict, enable: bool) -> dict:
        """
        Run wireframe, edge and line model for detection

        Args:
            items (dict): Input data
            enable (bool): Enable wireframe, edge and line model

        Returns:
            dict: return the items extracted from the wireframe, edge and line model
        """

        if not enable:
            items["edge"] = torch.zeros_like(items["masks"])
            items["line"] = torch.zeros_like(items["masks"])
            return

        start = time.time()

        try:
            line_256 = self.wireframe_forward(
                items["img_512"],
                h=256,
                w=256,
                masks=items["mask_512"],
                mask_th=0.85,
            )
        except:
            line_256 = torch.zeros_like(items["mask_256"])

        logger.debug(f"wireframe_forward time: {(time.time() - start) * 1000:.2f}ms")

        start = time.time()

        edge_pred, line_pred = self.sample_edge_line_logits(
            context=[items["img_256"], items["edge_256"], line_256],
            mask=items["mask_256"].clone(),
            iterations=self.sample_edge_line_iterations,
            add_v=0.05,
            mul_v=4,
        )
        logger.debug(
            f"sample_edge_line_logits time: {(time.time() - start) * 1000:.2f}ms"
        )

        input_size = min(items["h"], items["w"])
        if input_size != 256 and input_size > 256:
            while edge_pred.shape[2] < input_size:
                edge_pred = self.structure_upsample(edge_pred)
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)

                line_pred = self.structure_upsample(line_pred)
                line_pred = torch.sigmoid((line_pred + 2) * 2)

            edge_pred = F.interpolate(
                edge_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            line_pred = F.interpolate(
                line_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )

        items["edge"] = edge_pred.detach()
        items["line"] = line_pred.detach()

    @torch.no_grad()
    def forward(self, image, mask, config: Config) -> torch.Tensor:
        """
        Run the model
        Note: Input images and output images have same size

        Args:
            image (np.ndarray): Input image [H, W, C] RGB
            mask (np.ndarray): Input mask [H, W] 0-1
            config (Config): Config object with model parameters and settings (see schema.py)

        Returns:
            torch.Tensor: Output image [H, W, C] RGB
        """

        mask = mask[:, :, 0]
        items = load_image(image, mask, device=self.device)

        self.wireframe_edge_and_line(items, config.zits_wireframe)

        inpainted_image = self.inpaint(
            items["images"],
            items["masks"],
            items["edge"],
            items["line"],
            items["rel_pos"],
            items["direct"],
        )

        inpainted_image = inpainted_image * 255.0
        inpainted_image = (
            inpainted_image.cpu().permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        )
        inpainted_image = inpainted_image[:, :, ::-1]

        return inpainted_image

    def wireframe_forward(self, images, h, w, masks, mask_th=0.925) -> torch.Tensor:
        """
        Predict wireframe for the given images and masks using the wireframe model and return the wireframe

        Args:
            images (torch.Tensor): Input images [B, C, H, W] RGB
            h (int): Height of the wireframe to be predicted
            w (int): Width of the wireframe to be predicted
            masks (torch.Tensor): Input masks [B, 1, H, W] 0-1
            mask_th (float): Threshold for the mask to be considered as foreground

        Returns:
            torch.Tensor: Predicted wireframe [B, 1, H, W] 0-1
        """

        lcnn_mean = torch.tensor([109.730, 103.832, 98.681]).reshape(1, 3, 1, 1)
        lcnn_std = torch.tensor([22.275, 22.124, 23.229]).reshape(1, 3, 1, 1)
        images = images * 255.0
        # the masks value of lcnn is 127.5
        masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
        masked_images = (masked_images - lcnn_mean) / lcnn_std

        def to_int(x):
            return tuple(map(int, x))

        lines_tensor = []
        lmap = np.zeros((h, w))

        output_masked = self.wireframe(masked_images)

        output_masked = to_device(output_masked, "cpu")
        if output_masked["num_proposals"] == 0:
            lines_masked = []
            scores_masked = []
        else:
            lines_masked = output_masked["lines_pred"].numpy()
            lines_masked = [
                [line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                for line in lines_masked
            ]
            scores_masked = output_masked["lines_score"].numpy()

        for line, score in zip(lines_masked, scores_masked):
            if score > mask_th:
                rr, cc, value = skimage.draw.line_aa(
                    *to_int(line[0:2]), *to_int(line[2:4])
                )
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
        lines_tensor.append(to_tensor(lmap).unsqueeze(0))

        lines_tensor = torch.cat(lines_tensor, dim=0)

        return lines_tensor.detach().to(self.device)

    def sample_edge_line_logits(
        self,
        context: List[torch.Tensor],
        mask: torch.Tensor,
        iterations: int = 3,
        add_v: int = 0,
        mul_v: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample edge and line logits from context

        Args:
            context (List[torch.Tensor]): Context tensors [image, edge, line]
            mask (torch.Tensor): Mask [B, 1, H, W] 0-1
            iterations (int): Number of iterations to sample edge and line logits from context (default: 3)
            add_v (float): Add value to edge and line logits before sigmoid (default: 0)
            mul_v (float): Multiply value to edge and line logits before sigmoid (default: 4)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge and line logits
        """

        [img, edge, line] = context

        img = img * (1 - mask)
        edge = edge * (1 - mask)
        line = line * (1 - mask)

        for i in range(iterations):
            edge_logits, line_logits = self.edge_line(img, edge, line, masks=mask)

            edge_pred = torch.sigmoid(edge_logits)
            line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
            edge = edge + edge_pred * mask
            edge[edge >= 0.25] = 1
            edge[edge < 0.25] = 0
            line = line + line_pred * mask

            b, _, h, w = edge_pred.shape
            edge_pred = edge_pred.reshape(b, -1, 1)
            line_pred = line_pred.reshape(b, -1, 1)
            mask = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
            line_probs = torch.cat([1 - line_pred, line_pred], dim=-1)
            edge_probs[:, :, 1] += 0.5
            line_probs[:, :, 1] += 0.5
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)
            line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask) * (-100)

            indices = torch.sort(
                edge_max_probs + line_max_probs, dim=-1, descending=True
            )[1]

            for ii in range(b):
                keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))

                assert torch.sum(mask[ii][indices[ii, :keep]]) == keep, "Error!!!"
                mask[ii][indices[ii, :keep]] = 0

            mask = mask.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

        edge, line = edge.to(torch.float32), line.to(torch.float32)

        return edge, line

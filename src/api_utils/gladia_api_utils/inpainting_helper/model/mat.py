import os
import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..helper import get_cache_path_by_url, load_model, norm_img
from ..schema import Config
from .base import InpaintModel
from .utils import (
    Conv2dLayer,
    FullyConnectedLayer,
    MinibatchStdLayer,
    activation_funcs,
    bias_act,
    conv2d_resample,
    normalize_2nd_moment,
    setup_filter,
    to_2tuple,
    upsample2d,
)


class ModulatedConv2d(torch.nn.Module):
    """
    2d convolution with style modulation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Width and height of the convolution kernel.
        style_dim (int): dimension of the style code
        demodulate (bool, optional): perfrom demodulation. (default: True)
        up (int, optional): Integer upsampling factor. (default: 1)
        down (int, optional): Integer downsampling factor. (default: 1)
        resample_filter (List[int, int, int, int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
        conv_clamp (Union[float, None], optional): Clamp the output to +-X, None = disable clamping. (default: None)

    Inherited from torch.nn.Module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        up: int = 1,
        down: int = 1,
        resample_filter: List[int] = [
            1,
            3,
            3,
            1,
        ],
        conv_clamp: Union[float, None] = None,
    ) -> None:
        """
        Constructor for ModulatedConv2d class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Width and height of the convolution kernel.
            style_dim (int): dimension of the style code
            demodulate (bool, optional): perfrom demodulation. (default: True)
            up (int, optional): Integer upsampling factor. (default: 1)
            down (int, optional): Integer downsampling factor. (default: 1)
            resample_filter (List[int, int, int, int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
            conv_clamp (Union[float, None], optional): Clamp the output to +-X, None = disable clamping. (default: None)

        Returns:
            None
        """
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ModulatedConv2d.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            style (torch.Tensor): Style tensor of shape (N, style_dim).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch,
        )
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


class StyleConv(torch.nn.Module):
    """
    Style convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        style_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        kernel_size (int, optional): Convolution kernel size. (default: 3)
        up (int, optional): Integer upsampling factor. (default: 1)
        use_noise (bool, optional): Enable noise input? (default: False)
        activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
        resample_filter (List[int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
        conv_clamp (Union[float, None], optional): Clamp the output of convolution layers to +-X, None = disable clamping. (default: None)
        demodulate (bool, optional): perform demodulation. (default: True)

    Inherited from torch.nn.Module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        resolution: int,
        kernel_size: int = 3,
        up: int = 1,
        use_noise: bool = False,
        activation: str = "lrelu",
        resample_filter: List[int] = [1, 3, 3, 1],
        conv_clamp: Union[float, None] = None,
        demodulate: bool = True,
    ):
        """
        Constructor for StyleConv class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            style_dim (int): Intermediate latent (W) dimensionality.
            resolution (int): Resolution of this layer.
            kernel_size (int, optional): Convolution kernel size. (default: 3)
            up (int, optional): Integer upsampling factor. (default: 1)
            use_noise (bool, optional): Enable noise input? (default: False)
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
            resample_filter (List[int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
            conv_clamp (Union[float, None], optional): Clamp the output of convolution layers to +-X, None = disable clamping. (default: None)
            demodulate (bool, optional): perform demodulation. (default: True)

        Returns:
            None
        """
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            up=up,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise_mode: str = "random",
        gain: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass of StyleConv.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            style (torch.Tensor): Style tensor of shape (N, style_dim).
            noise_mode (str, optional): Noise mode: 'random', 'const', 'none'. (default: 'random')
            gain (float, optional): Overall scaling factor. (default: 1.0)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        x = self.conv(x, style)

        assert noise_mode in ["random", "const", "none"]

        if self.use_noise:
            if noise_mode == "random":
                xh, xw = x.size()[-2:]
                noise = (
                    torch.randn([x.shape[0], 1, xh, xw], device=x.device)
                    * self.noise_strength
                )
            if noise_mode == "const":
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )

        return out


class ToRGB(torch.nn.Module):
    """
    ToRGB layer. Converts convolution output to RGB image.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        style_dim (int): Intermediate latent (W) dimensionality.
        kernel_size (int, optional): Convolution kernel size. (default: 1)
        resample_filter (List[int, int, int, int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
        conv_clamp (Union[float, None], optional): Clamp the output of convolution layers to +-X, None = disable clamping. (default: None)
        demodulate (bool, optional): perform demodulation. (default: False)

    Inherited from torch.nn.Module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 1,
        resample_filter: List[int] = [1, 3, 3, 1],
        conv_clamp=None,
        demodulate=False,
    ) -> None:
        """
        Constructor for ToRGB class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            style_dim (int): Intermediate latent (W) dimensionality.
            kernel_size (int, optional): Convolution kernel size. (default: 1)
            resample_filter (List[int, int, int, int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
            conv_clamp (Union[float, None], optional): Clamp the output of convolution layers to +-X, None = disable clamping. (default: None)
            demodulate (bool, optional): perform demodulation. (default: False)

        Returns:
            None
        """

        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        skip: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Forward pass of ToRGB.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            style (torch.Tensor): Style tensor of shape (N, style_dim).
            skip (torch.Tensor, optional): Skip tensor of shape (N, C, H, W). (default: None)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        x = self.conv(x, style)
        out = bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upsample2d(skip, self.resample_filter)
            out = out + skip

        return out


def get_style_code(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get style code from two images.

    Args:
        a (torch.Tensor): Image tensor of shape (N, C, H, W).
        b (torch.Tensor): Image tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Style code tensor of shape (N, style_dim).
    """
    return torch.cat([a, b], dim=1)


class DecBlockFirst(torch.nn.Module):
    """
    Decoder block for the first block.

    Args:


    Inherited from torch.nn.Module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for DecBlockFirst class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function: 'relu', 'lrelu'.
            style_dim (int): Intermediate latent (W) dimensionality.
            use_noise (bool): Enable noise input.
            demodulate (bool): perform demodulation.
            img_channels (int): Number of output image channels.

        Returns:
            None
        """
        super().__init__()
        self.fc = FullyConnectedLayer(
            in_features=in_channels * 2,
            out_features=in_channels * 4**2,
            activation=activation,
        )
        self.conv = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=4,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward pass of DecBlockFirst.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            ws (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            gs (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            E_features (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            noise_mode (str, optional): Noise mode: 'random', 'const'. (default: 'random')

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


class DecBlockFirstV2(torch.nn.Module):
    """
    Decoder block for the first block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function: 'relu', 'lrelu'.
        style_dim (int): Intermediate latent (W) dimensionality.
        use_noise (bool): Enable noise input.
        demodulate (bool): perform demodulation.
        img_channels (int): Number of output image channels.

    Inherited from torch.nn.Module

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for DecBlockFirst class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function: 'relu', 'lrelu'.
            style_dim (int): Intermediate latent (W) dimensionality.
            use_noise (bool): Enable noise input.
            demodulate (bool): perform demodulation.
            img_channels (int): Number of output image channels.

        Returns:
            None
        """
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=4,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ):
        """
        Forward pass of DecBlockFirst.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            ws (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            gs (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            E_features (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            noise_mode (str, optional): Noise mode: 'random', 'const'. (default: 'random')

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


class DecBlock(torch.nn.Module):
    """
    Class for decoder block.

    Args:
        res (int): Resolution of the block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function: 'relu', 'lrelu'.
        style_dim (int): Intermediate latent (W) dimensionality.
        use_noise (bool): Enable noise input.
        demodulate (bool): perform demodulation.
        img_channels (int): Number of output image channels.

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        res: int,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for DecBlock class.

        Args:
            res (int): Resolution of the block.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function: 'relu', 'lrelu'.
            style_dim (int): Intermediate latent (W) dimensionality.
            use_noise (bool): Enable noise input.
            demodulate (bool): perform demodulation.
            img_channels (int): Number of output image channels.

        Returns:
            None
        """
        # res = 2, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        img: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ):
        """
        Fopward pass of DecBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            img (torch.Tensor): Input tensor of shape (N, C, H, W).
            ws (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            gs (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            E_features (torch.Tensor): Style tensor of shape (N, num_layers, style_dim).
            noise_mode (str, optional): Noise mode: 'random', 'const'. (default: 'random')

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        style = get_style_code(ws[:, self.res * 2 - 5], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 4], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 3], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


class MappingNet(torch.nn.Module):
    """
    Class for mapping network.

    Args:
        z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        num_ws (int): Number of intermediate latents to output, None = do not broadcast.
        num_layers (int, optional): Number of mapping layers. (default: 8)
        embed_features (Union[int, None], optional): Label embedding dimensionality, None = same as w_dim. (default: None)
        layer_features (Union[int, None], optional): Number of intermediate features in the mapping layers, None = same as w_dim. (default: None) (unused)
        activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
        lr_multiplier (float, optional): Learning rate multiplier for the mapping layers. (default: 0.01)
        w_avg_beta (Union[float, None], optional): Decay for tracking the moving average of W during training, None = do not track. (default: 0.995)

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        num_ws: int,
        num_layers: int = 8,
        embed_features: Union[int, None] = None,
        layer_features: Union[int, None] = None,
        activation: str = "lrelu",
        lr_multiplier: float = 0.01,
        w_avg_beta: Union[float, None] = 0.995,
    ) -> None:
        """
        Constructor for MappingNet class.

        Args:
            z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
            c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
            w_dim (int): Intermediate latent (W) dimensionality.
            num_ws (int): Number of intermediate latents to output, None = do not broadcast.
            num_layers (int, optional): Number of mapping layers. (default: 8)
            embed_features (Union[int, None], optional): Label embedding dimensionality, None = same as w_dim. (default: None)
            layer_features (Union[int, None], optional): Number of intermediate features in the mapping layers, None = same as w_dim. (default: None)
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
            lr_multiplier (float, optional): Learning rate multiplier for the mapping layers. (default: 0.01)
            w_avg_beta (Union[float, None], optional): Decay for tracking the moving average of W during training, None = do not track. (default: 0.995)

        Returns:
            None
        """
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        truncation_psi: float = 1,
        truncation_cutoff: Union[int, None] = None,
        skip_w_avg_update: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of MappingNet.

        Args:
            z (torch.Tensor): Input tensor of shape (N, z_dim).
            c (torch.Tensor): Input tensor of shape (N, c_dim).
            truncation_psi (float, optional): Truncation psi. (default: 1.0)
            truncation_cutoff (Union[int, None], optional): Truncation cutoff. (default: None)
            skip_w_avg_update (bool, optional): Skip w_avg update. (default: False)

        Returns:
            torch.Tensor: Output tensor of shape (N, num_ws, w_dim).

        """
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )

        return x


class DisFromRGB(torch.nn.Module):
    """
    Class for the fromRGB layer of the discriminator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function: 'relu', 'lrelu', etc.

    Inherits from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        """
        Constructor for DisFromRGB class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function: 'relu', 'lrelu', etc.

        Returns:
            None
        """
        # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DisFromRGB.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        return self.conv(x)


class DisBlock(torch.nn.Module):
    """
    Class for a block of the discriminator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function: 'relu', 'lrelu', etc.

    Inherits from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        """
        Constructor for DisBlock class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function: 'relu', 'lrelu', etc.

        Returns:
            None
        """
        # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=2,
            activation=activation,
        )
        self.skip = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            down=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DisBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x

        return out


class Discriminator(torch.nn.Module):
    """
    Class for the discriminator.

    Args:
        c_dim (int): Conditioning label (C) dimensionality.
        img_resolution (int): Input resolution.
        img_channels (int): Number of input color channels.
        channel_base (int): Overall multiplier for the number of channels. (default: 32768)
        channel_max (int): Maximum number of channels in any layer. (default: 512)
        channel_decay (int): Log2 channel reduction when doubling the resolution. (default: 1)
        cmap_dim (Union[int, None]): Dimensionality of mapped conditioning label, None = default. (default: None)
        activation (str): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
        mbstd_group_size (int): Group size for the minibatch standard deviation layer, None = entire minibatch. (default: 4)
        mbstd_num_channels (int): Number of features for the minibatch standard deviation layer, 0 = disable. (default: 1)

    Inherits from `torch.nn.Module`.
    """

    def __init__(
        self,
        c_dim: int,
        img_resolution: int,
        img_channels: int,
        channel_base: int = 32768,
        channel_max: int = 512,
        channel_decay: int = 1,
        cmap_dim: Union[int, None] = None,
        activation: str = "lrelu",
        mbstd_group_size: int = 4,
        mbstd_num_channels: int = 1,
    ) -> None:
        """
        Constructor for Discriminator class.

        Args:
            c_dim (int): Conditioning label (C) dimensionality.
            img_resolution (int): Input resolution.
            img_channels (int): Number of input color channels.
            channel_base (int): Overall multiplier for the number of channels. (default: 32768)
            channel_max (int): Maximum number of channels in any layer. (default: 512)
            channel_decay (int): Log2 channel reduction when doubling the resolution. (default: 1)
            cmap_dim (Union[int, None]): Dimensionality of mapped conditioning label, None = default. (default: None)
            activation (str): Activation function: 'relu', 'lrelu', etc. (default: 'lrelu')
            mbstd_group_size (int): Group size for the minibatch standard deviation layer, None = entire minibatch. (default: 4)
            mbstd_num_channels (int): Number of features for the minibatch standard deviation layer, 0 = disable. (default: 1)

        Returns:
            None
        """
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        def nf(stage):
            return np.clip(
                int(channel_base / 2 ** (stage * channel_decay)), 1, channel_max
            )

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None
            )

        Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res - 1), activation))

        if mbstd_num_channels > 0:
            Dis.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis.append(
            Conv2dLayer(
                nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation
            )
        )
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2) * 4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

    def forward(
        self, images_in: torch.Tensor, masks_in: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of Discriminator.

        Args:
            images_in (torch.Tensor): Input tensor of shape (N, img_channels, H, W).
            masks_in (torch.Tensor): Input tensor of shape (N, 1, H, W).
            c (torch.Tensor): Input tensor of shape (N, c_dim).

        Returns:
            torch.Tensor: Output tensor of shape (N, 1) if cmap_dim == 0 else (N, cmap_dim).
        """
        x = torch.cat([masks_in - 0.5, images_in], dim=1)
        x = self.Dis(x)
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x


def nf(
    stage: int,
    channel_base: int = 32768,
    channel_decay: float = 1.0,
    channel_max: int = 512,
) -> int:
    """
    Number of filters in a given stage.

    Args:
        stage (int): Stage.
        channel_base (int, optional): Overall multiplier for the number of channels. (default: 32768) (unused)
        channel_decay (float, optional): log2 channel reduction when doubling the resolution. (default: 1.0) (unused)
        channel_max (int, optional): Maximum number of channels in any layer. (default: 512) (unused)

    Returns:
        int: Number of filters.
    """
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2**stage]


class Mlp(torch.nn.Module):
    """
    Class for MLP. (Multi-layer perceptron)

    Args:
        in_features (int): Number of input features.
        hidden_features (Union[int, None], optional): Number of hidden features. (default: None)
        out_features (Union[int, None], optional): Number of output features. (default: None)
        act_layer (torch.nn.Module, optional): Activation layer. (default: nn.GELU) (unused)
        drop (float, optional): Dropout probability. (default: 0.0) (unused)

    Inherits from `torch.nn.Module`.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Union[int, None] = None,
        out_features: Union[int, None] = None,
        act_layer: torch.nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """
        Constructor for Mlp class.

        Args:
            in_features (int): Number of input features.
            hidden_features (Union[int, None], optional): Number of hidden features. (default: None)
            out_features (Union[int, None], optional): Number of output features. (default: None)
            act_layer (torch.nn.Module, optional): Activation layer. (default: nn.GELU) (unused)
            drop (float, optional): Dropout probability. (default: 0.0) (unused)

        Returns:
            None
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(
            in_features=in_features, out_features=hidden_features, activation="lrelu"
        )
        self.fc2 = FullyConnectedLayer(
            in_features=hidden_features, out_features=out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mlp.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_features).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # B = windows.shape[0] / (H * W / window_size / window_size)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Conv2dLayerPartial(torch.nn.Module):
    """
    Conv2d layer with partial convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Width and height of the convolution kernel.
        bias (bool, optional): Apply additive bias before the activation function? (default: True)
        activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'linear')
        up (int, optional): Integer upsampling factor. (default: 1)
        down (int, optional): Integer downsampling factor. (default: 1)
        resample_filter (List[int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
        conv_clamp (Union[float, None], optional): Clamp the output to +-X, None = disable clamping. (default: None)
        trainable (bool, optional): Update the weights of this layer during training? (default: True)

    Inherited from `torch.nn.Module`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: str = "linear",
        up: int = 1,
        down: int = 1,
        resample_filter: List[int] = [
            1,
            3,
            3,
            1,
        ],
        conv_clamp: Union[float, None] = None,
        trainable: bool = True,
    ):
        """
        Construct a Conv2dLayer object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Width and height of the convolution kernel.
            bias (bool, optional): Apply additive bias before the activation function? (default: True)
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: 'linear')
            up (int, optional): Integer upsampling factor. (default: 1)
            down (int, optional): Integer downsampling factor. (default: 1)
            resample_filter (List[int], optional): Low-pass filter to apply when resampling activations. (default: [1, 3, 3, 1])
            conv_clamp (Union[float, None], optional): Clamp the output to +-X, None = disable clamping. (default: None)
            trainable (bool, optional): Update the weights of this layer during training? (default: True)

        Returns:
            None
        """
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            activation,
            up,
            down,
            resample_filter,
            conv_clamp,
            trainable,
        )

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size**2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(
        self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass of Conv2dLayerPartial.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
            mask (Union[torch.Tensor, None], optional): Mask tensor of shape (N, 1, H, W). (default: None)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """

        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


class WindowAttention(torch.nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        down_ratio (int, optional): Downsample the feature map by a factor of down_ratio (default: 1). (unused)
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. (default: True) (unused)
        qk_scale ([Union[float, None]], optional): Override default qk scale of head_dim ** -0.5 if set. (default: None)
        attn_drop (float, optional): Dropout ratio of attention weight. (default: 0.0) (unused)
        proj_drop (float, optional): Dropout ratio of output. (default: 0.0) (unused)

    Inherited from torch.nn.Module.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        down_ratio: int = 1,
        qkv_bias: bool = True,
        qk_scale: Union[float, None] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Constructor method for WindowAttention.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            down_ratio (int, optional): Downsample the feature map by a factor of down_ratio (default: 1). (unused)
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. (default: True) (unused)
            qk_scale ([Union[float, None]], optional): Override default qk scale of head_dim ** -0.5 if set. (default: None)
            attn_drop (float, optional): Dropout ratio of attention weight. (default: 0.0) (unused)
            proj_drop (float, optional): Dropout ratio of output. (default: 0.0) (unused)

        Returns:
            None
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask_windows: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Forward function for W-MSA.

        Args:
            x (torch.Tensor): Input feature with shape of (num_windows*B, N, C)
            mask_windows (Union[torch.Tensor, None], optional): 2D mask with shape of (num_windows, window_size*window_size),
                value should be between (-inf, 0]. Only non-zero values are allowed. (default: None)
            mask (Union[torch.Tensor, None], optional):(0/-inf) 2D mask with shape of (num_windows*B, Wh*Ww) (default: None)

        Returns:
            torch.Tensor: Output feature with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)
            ).masked_fill(attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(
                    torch.sum(mask_windows, dim=1, keepdim=True), 0, 1
                ).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows


class SwinTransformerBlock(torch.nn.Module):
    """
    Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        down_ratio (int, optional): Down ratio for patch merging. (default: 1)
        window_size (int, optional): Window size. (default: 7)
        shift_size (int, optional): Shift size for SW-MSA. (default: 0)
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. (default: 4.0)
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (default: True)
        qk_scale (Union[float, None], optional): Override default qk scale of head_dim ** -0.5 if set. (default: None)
        drop (float, optional): Dropout rate. (default: 0.0)
        attn_drop (float, optional): Attention dropout rate. (default: 0.0)
        drop_path (float, optional): Stochastic depth rate. (default: 0.0) (unused)
        act_layer (torch.nn.Module, optional): Activation layer. (default: nn.GELU)
        norm_layer (torch.nn.Module, optional): Normalization layer. (default: nn.LayerNorm)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        down_ratio: int = 1,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Union[float, None] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: torch.nn.Module = nn.GELU,
        norm_layer: torch.nn.Module = nn.LayerNorm,
    ) -> None:
        """
        Constructor method for SwinTransformerBlock class.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resulotion.
            num_heads (int): Number of attention heads.
            down_ratio (int, optional): Down ratio for patch merging. (default: 1)
            window_size (int, optional): Window size. (default: 7)
            shift_size (int, optional): Shift size for SW-MSA. (default: 0)
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. (default: 4.0)
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (default: True)
            qk_scale (Union[float, None], optional): Override default qk scale of head_dim ** -0.5 if set. (default: None)
            drop (float, optional): Dropout rate. (default: 0.0)
            attn_drop (float, optional): Attention dropout rate. (default: 0.0)
            drop_path (float, optional): Stochastic depth rate. (default: 0.0) (unused)
            act_layer (torch.nn.Module, optional): Activation layer. (default: nn.GELU)
            norm_layer (torch.nn.Module, optional): Normalization layer. (default: nn.LayerNorm)

        Returns:
            None
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.fuse = FullyConnectedLayer(
            in_features=dim * 2, out_features=dim, activation="lrelu"
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        Calculate 2D attention mask for SW-MSA.

        Args:
            x_size (Tuple[int, int]): Input resulotion.

        Returns:
            torch.Tensor: 2D attention mask.
        """
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Forward function for `SwinTransformerBlock`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C).
            x_size (Tuple[int, int]): Input resulotion.
            mask (Union[torch.Tensor, None], optional): Attention mask. (default: None)

        Returns:
            torch.Tensor: Output of this block.
        """

        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                shifted_mask = torch.roll(
                    mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
            )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                mask = torch.roll(
                    shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask


class PatchMerging(torch.nn.Module):
    """
    Class for patch merging.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        down (int, optional): Downsample factor. (default: 2)

    Inherited from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, down: int = 2) -> None:
        """
        Constructor for `PatchMerging`.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            down (int, optional): Downsample factor. (default: 2)

        Returns:
            None
        """
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            down=down,
        )
        self.down = down

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Forward function for `PatchMerging`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C).
            x_size (Tuple[int, int]): Input resulotion.
            mask (Union[torch.Tensor, None], optional): Attention mask. (default: None)

        Returns:
            torch.Tensor: Output of this block.
        """
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class PatchUpsampling(torch.nn.Module):
    """
    Class for patch upsampling.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        up (int, optional): Upsample factor. (default: 2)

    Inherited from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, up: int = 2) -> None:
        """
        Constructor for `PatchUpsampling`.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            up (int, optional): Upsample factor. (default: 2)

        Returns:
            None
        """
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            up=up,
        )
        self.up = up

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Forward function for `PatchUpsampling`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C).
            x_size (Tuple[int, int]): Input resulotion.
            mask (Union[torch.Tensor, None], optional): Attention mask. (default: None)

        Returns:
            torch.Tensor: Output of this block.
        """
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class BasicLayer(torch.nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (Union[float, None], optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. (default: 0.0)
        attn_drop (float, optional): Attention dropout rate. (default: 0.0)
        drop_path (Union[float, tuple[float]], optional): Stochastic depth rate. (default: 0.0)
        norm_layer (torch.nn.Module, optional): Normalization layer. (default: nn.LayerNorm)
        downsample (Union[torch.nn.Module, None, optional): Downsample layer at the end of the layer. (default: None)
        use_checkpoint (bool): Whether to use checkpointing to save memory. (default: False)

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        down_ratio=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ) -> None:
        """
        Constructor for `BasicLayer`.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (Union[float, None], optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. (default: 0.0)
            attn_drop (float, optional): Attention dropout rate. (default: 0.0)
            drop_path (Union[float, tuple[float]], optional): Stochastic depth rate. (default: 0.0)
            norm_layer (torch.nn.Module, optional): Normalization layer. (default: nn.LayerNorm)
            downsample (Union[torch.nn.Module, None, optional): Downsample layer at the end of the layer. (default: None)
            use_checkpoint (bool): Whether to use checkpointing to save memory. (default: False)

        Returns:
            None
        """

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    down_ratio=down_ratio,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation="lrelu"
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Forward function for `BasicLayer`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C).
            x_size (Tuple[int, int]): Input resulotion.
            mask (Union[torch.Tensor, None], optional): Attention mask. (default: None)

        Returns:
            torch.Tensor: Output of this block.
        """
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class ToToken(torch.nn.Module):
    """
    Class to convert feature map to tokens.

    Args:
        in_channels (int, optional): Number of input channels. (default: 3)
        dim (int, optional): Number of output channels. (default: 128)
        kernel_size (int, optional): Kernel size of the conv layer. (default: 5)
        stride (int, optional): Stride of the conv layer. (default: 1) (unused)

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        kernel_size: int = 5,
        stride: int = 1,
    ) -> None:
        """
        Convert image to tokens.

        Args:
            in_channels (int, optional): Number of input channels. (default: 3)
            dim (int, optional): Number of output channels. (default: 128)
            kernel_size (int, optional): Kernel size of the conv layer. (default: 5)
            stride (int, optional): Stride of the conv layer. (default: 1) (unused)

        Returns:
            None
        """
        super().__init__()

        self.proj = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=kernel_size,
            activation="lrelu",
        )

    def forward(
        self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward function for `ToToken`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            mask (Union[torch.Tensor, None], optional): Attention mask. (default: None)

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: Output of this block.
        """
        x, mask = self.proj(x, mask)

        return x, mask


class EncFromRGB(torch.nn.Module):
    """
    Class for encoding image to tokens from RGB.

    Args:
        in_channels (int): Number of input channels.
        dim (int): Number of output channels.
        activation (str, optional): Activation function.

    Inherited from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        """
        Convert RGB image to feature map.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.

        Returns:
            None.
        """
        # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for `EncFromRGB`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of this block.
        """
        x = self.conv0(x)
        x = self.conv1(x)

        return x


class ConvBlockDown(torch.nn.Module):
    """
    Class for convolutional block with downsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function.

    Inherited from `torch.nn.Module`.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str):
        """
        Convolutional block for downsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.

        Returns:
            torch.Tensor: Output of this block.
        """
        # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for `ConvBlockDown`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of this block.
        """
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
    """
    Transform token to feature map.

    Args:
        x (torch.Tensor): Input tensor with shape (B, N, C).
        x_size (Tuple[int, int]): Input resulotion.

    Returns:
        torch.Tensor: Output of this block.
    """
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x: torch.Tensor) -> torch.Tensor:
    """
    Transform feature map to token.

    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W).

    Returns:
        torch.Tensor: Output of this block.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


class Encoder(torch.nn.Module):
    """
    Class for Encoder.

    Args:
        res_log2 (int): Resolution of the input image.
        img_channels (int): Number of channels of the input image.
        activation (str): Activation function.
        patch_size (int, optional): Patch size. (default: 5) (unused)
        channels (int, optional): Number of channels. (default: 16) (unused)
        drop_path_rate (float, optional): Drop path rate. (default: 0.1) (unused)

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        res_log2: int,
        img_channels: int,
        activation: str,
        patch_size: int = 5,
        channels: int = 16,
        drop_path_rate: float = 0.1,
    ) -> None:
        """
        Constructor for `Encoder`.

        Args:
            res_log2 (int): Resolution of the input image.
            img_channels (int): Number of channels of the input image.
            activation (str): Activation function.
            patch_size (int, optional): Patch size. (default: 5) (unused)
            channels (int, optional): Number of channels. (default: 16) (unused)
            drop_path_rate (float, optional): Drop path rate. (default: 0.1) (unused)

        Returns:
            None
        """
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2**i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, "EncConv_Block_%dx%d" % (res, res), block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for `Encoder`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of this block.
        """
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, "EncConv_Block_%dx%d" % (res, res))(x)
            out[res_log2] = x

        return out


class ToStyle(torch.nn.Module):
    """
    Class for `ToStyle` block. This block is used to transform token to style.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function.
        drop_rate (float): Dropout rate. (unused)

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, activation: str, drop_rate: float
    ) -> None:
        """
        Constructor method for `ToStyle`.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.
            drop_rate (float): Dropout rate. (unused)

        Returns:
            None
        """
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(
            in_features=in_channels, out_features=out_channels, activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for `ToStyle`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of this block.
        """

        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x


class DecBlockFirstV2(torch.nn.Module):
    """
    Class for the first block of the decoder.

    Args:
        res (int): Resolution of this block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function.
        style_dim (int): Style dimension.
        use_noise (bool): Whether to use noise.
        demodulate (bool): Whether to demodulate.
        img_channels (int): Number of image channels.

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        res: int,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ):
        """
        Constructor for `DecBlockFirstV2`.

        Args:
            res (int): Resolution of this block.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.
            style_dim (int): Style dimension.
            use_noise (bool): Whether to use noise.
            demodulate (bool): Whether to demodulate.
            img_channels (int): Number of image channels.

        Returns:
            None
        """
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward function for `DecBlockFirstV2`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            ws (torch.Tensor): Style tensor with shape (B, num_ws, C).
            gs (torch.Tensor): Style tensor with shape (B, num_gs, C).
            E_features (torch.Tensor): Feature tensor from encoder with shape (B, C, H, W).
            noise_mode (str): Noise mode. Defaults to "random".

        Returns:
            torch.Tensor: Output of this block.
        """
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


class DecBlock(torch.nn.Module):
    """
    Class for decoder block. It is used to upsample the image.

    Args:
        res (int): Resolution of this block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function.
        style_dim (int): Style dimension.
        use_noise (bool): Whether to use noise.
        demodulate (bool): Whether to demodulate.

    Inherited from `torch.nn.Module`.
    """

    def __init__(
        self,
        res: int,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for `DecBlock`.

        Args:
            res (int): Resolution of this block.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.
            style_dim (int): Style dimension.
            use_noise (bool): Whether to use noise.
            demodulate (bool): Whether to demodulate.

        Returns:
            None
        """

        # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        img: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward function for `DecBlock`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            img (torch.Tensor): Input tensor with shape (B, C, H, W).
            ws (torch.Tensor): Style tensor with shape (B, num_ws, C).
            gs (torch.Tensor): Style tensor with shape (B, num_gs, C).
            E_features (torch.Tensor): Feature tensor from encoder with shape (B, C, H, W).
            noise_mode (str): Noise mode. Defaults to "random".

        Returns:
            torch.Tensor: Output of this block.
        """
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


class Decoder(torch.nn.Module):
    """
    Class for the decoder

    Args:
        res_log2 (int): Resolution log2.
        activation (str): Activation function.
        style_dim (int): Style dimension.
        use_noise (bool): Whether to use noise.
        demodulate (bool): Whether to demodulate.
        img_channels (int): Number of image channels.

    Inherits from `torch.nn.Module`.
    """

    def __init__(
        self,
        res_log2: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for `Decoder`.

        Args:
            res_log2 (int): Resolution log2.
            activation (str): Activation function.
            style_dim (int): Style dimension.
            use_noise (bool): Whether to use noise.
            demodulate (bool): Whether to demodulate.
            img_channels (int): Number of image channels.

        Returns:
            None
        """
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(
            4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels
        )
        for res in range(5, res_log2 + 1):
            setattr(
                self,
                "Dec_%dx%d" % (2**res, 2**res),
                DecBlock(
                    res,
                    nf(res - 1),
                    nf(res),
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                ),
            )
        self.res_log2 = res_log2

    def forward(
        self,
        x: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor,
        E_features: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward function for `Decoder`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            ws (torch.Tensor): Style tensor with shape (B, num_ws, C).
            gs (torch.Tensor): Style tensor with shape (B, num_gs, C).
            E_features (torch.Tensor): Feature tensor from encoder with shape (B, C, H, W).
            noise_mode (str): Noise mode. Defaults to "random".

        Returns:
            torch.Tensor: Output of this block.
        """
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, "Dec_%dx%d" % (2**res, 2**res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


class DecStyleBlock(torch.nn.Module):
    """
    Class for decoder style block. This block is used to generate style codes for the decoder.

    Args:
        res (int): Resolution.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function.
        style_dim (int): Style dimension.
        use_noise (bool): Whether to use noise.
        demodulate (bool): Whether to demodulate.
        img_channels (int): Number of image channels.

    Inherits from `torch.nn.Module`.
    """

    def __init__(
        self,
        res: int,
        in_channels: int,
        out_channels: int,
        activation: str,
        style_dim: int,
        use_noise: bool,
        demodulate: bool,
        img_channels: int,
    ) -> None:
        """
        Constructor for `DecStyleBlock`.


        Args:
            res (int): Resolution.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.
            style_dim (int): Style dimension.
            use_noise (bool): Whether to use noise.
            demodulate (bool): Whether to demodulate.
            img_channels (int): Number of image channels.

        Returns:
            None
        """
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        img: torch.Tensor,
        style: torch.Tensor,
        skip: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward function for `DecBlock`.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            img (torch.Tensor): Input tensor with shape (B, C, H, W).
            style (torch.Tensor): Style tensor with shape (B, num_ws, C).
            skip (torch.Tensor): Skip tensor with shape (B, C, H, W).
            noise_mode (str): Noise mode. Defaults to "random".

        Returns:
            torch.Tensor: Output of this block.
        """

        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


class FirstStage(torch.nn.Module):
    """
    Class for the first stage of the generator.

    Args:
        img_channels (int): Number of channels in the image.
        img_resolution (int): Resolution of the image. (default: 256)
        dim (int): Dimension of the style code. (default: 180)
        w_dim (int): Dimension of the style code. (default: 512)
        use_noise (bool): Whether to use noise in the convolution. (default: False)
        demodulate (bool): Whether to use demodulation in the convolution. (default: True)
        activation (str): Activation function. (default: "lrelu")
    """

    def __init__(
        self,
        img_channels: int,
        img_resolution: int = 256,
        dim: int = 180,
        w_dim: int = 512,
        use_noise: bool = False,
        demodulate: bool = True,
        activation: str = "lrelu",
    ) -> None:
        """
        Constructor for `FirstStage`.

        Args:
            img_channels (int): Number of channels in the image.
            img_resolution (int): Resolution of the image. (default: 256)
            dim (int): Dimension of the style code. (default: 180)
            w_dim (int): Dimension of the style code. (default: 512)
            use_noise (bool): Whether to use noise in the convolution. (default: False)
            demodulate (bool): Whether to use demodulation in the convolution. (default: True)
            activation (str): Activation function. (default: "lrelu")

        Returns:
            None
        """
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels + 1,
            out_channels=dim,
            kernel_size=3,
            activation=activation,
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        # 根据图片尺寸构建 swim transformer 的层数
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=[res, res],
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=merge,
                )
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(
            in_features=dim, out_features=dim * 2, activation=activation
        )
        self.ws_style = FullyConnectedLayer(
            in_features=w_dim, out_features=dim, activation=activation
        )
        self.to_square = FullyConnectedLayer(
            in_features=dim, out_features=16 * 16, activation=activation
        )

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(
                DecStyleBlock(
                    res,
                    dim,
                    dim,
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                )
            )

    def forward(
        self,
        images_in: torch.Tensor,
        masks_in: torch.Tensor,
        ws: torch.Tensor,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """
        Forward function for `FirstStage`.

        Args:
            images_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            masks_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            ws (torch.Tensor): Style tensor with shape (B, num_ws, C).
            noise_mode (str): Noise mode. Defaults to "random".

        Returns:
            torch.Tensor: Output of this block.
        """
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                add_n = self.to_square(ws).unsqueeze(1)
                add_n = (
                    F.interpolate(
                        add_n, size=x.size(1), mode="linear", align_corners=False
                    )
                    .squeeze(1)
                    .unsqueeze(-1)
                )
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                )
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv):
            x, img = block(
                x, img, style, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode
            )

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


class SynthesisNet(torch.nn.Module):
    """
    Class for synthesis network. This network is used to generate images from
    style codes.

    Args:
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int, optional): Number of color channels. (default: 3)
        channel_base (int, optional): Overall multiplier for the number of channels. (default: 32768)
        channel_decay (float, optional): log2 channel multiplier decay per resolution. (default: 1.0)
        channel_max (int, optional): Maximum number of channels in any layer. (default: 512)
        activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: "lrelu")
        drop_rate (float, optional): Dropout rate. (default: 0.5)
        use_noise (bool, optional): Whether to use noise. (default: False)
        demodulate (bool, optional): Whether to use demodulation. (default: True)
    """

    def __init__(
        self,
        w_dim: int,
        img_resolution: int,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_decay: float = 1.0,
        channel_max: int = 512,
        activation: str = "lrelu",
        drop_rate: float = 0.5,
        use_noise: bool = False,
        demodulate: bool = True,
    ):
        """
        Constructor for `SynthesisNet`.

        Args:
            w_dim (int): Intermediate latent (W) dimensionality.
            img_resolution (int): Output image resolution.
            img_channels (int, optional): Number of color channels. (default: 3)
            channel_base (int, optional): Overall multiplier for the number of channels. (default: 32768)
            channel_decay (float, optional): log2 channel multiplier decay per resolution. (default: 1.0)
            channel_max (int, optional): Maximum number of channels in any layer. (default: 512)
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. (default: "lrelu")
            drop_rate (float, optional): Dropout rate. (default: 0.5)
            use_noise (bool, optional): Whether to use noise. (default: False)
            demodulate (bool, optional): Whether to use demodulation. (default: True)

        Returns:
            None
        """
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(
            img_channels,
            img_resolution=img_resolution,
            w_dim=w_dim,
            use_noise=False,
            demodulate=demodulate,
        )

        # second stage
        self.enc = Encoder(
            resolution_log2, img_channels, activation, patch_size=5, channels=16
        )
        self.to_square = FullyConnectedLayer(
            in_features=w_dim, out_features=16 * 16, activation=activation
        )
        self.to_style = ToStyle(
            in_channels=nf(4),
            out_channels=nf(2) * 2,
            activation=activation,
            drop_rate=drop_rate,
        )
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(
            resolution_log2, activation, style_dim, use_noise, demodulate, img_channels
        )

    def forward(
        self,
        images_in: torch.Tensor,
        masks_in: torch.Tensor,
        ws: torch.Tensor,
        noise_mode: str = "random",
        return_stg1: bool = False,
    ) -> torch.Tensor:
        """
        Forward function for `SynthesisNet`.

        Args:
            images_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            masks_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            ws (torch.Tensor): Style tensor with shape (B, num_ws, C).
            noise_mode (str): Noise mode. Defaults to "random".
            return_stg1 (bool): Whether to return the output of the first stage. Defaults to False.

        Returns:
            torch.Tensor: Output of this block.
        """
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode="bilinear", align_corners=False
        )
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        if not return_stg1:
            return img
        else:
            return img, out_stg1


class Generator(torch.nn.Module):
    """
    Class Generator.

    Args:
        z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): resolution of generated image
        img_channels (int): Number of input color channels.
        synthesis_kwargs (dict, optional): Arguments for SynthesisNetwork. (default: {})
        mapping_kwargs (dict, optional): Arguments for MappingNetwork. (default: {})
    """

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        img_resolution: int,
        img_channels: int,
        synthesis_kwargs: dict = {},
        mapping_kwargs: dict = {},
    ) -> None:
        """
        Constructor for `Generator`.

        Args:
            z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
            c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
            w_dim (int): Intermediate latent (W) dimensionality.
            img_resolution (int): resolution of generated image
            img_channels (int): Number of input color channels.
            synthesis_kwargs (dict, optional): Arguments for SynthesisNetwork. (default: {})
            mapping_kwargs (dict, optional): Arguments for MappingNetwork. (default: {})

        Returns:
            None
        """
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.mapping = MappingNet(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_layers,
            **mapping_kwargs,
        )

    def forward(
        self,
        images_in: torch.Tensor,
        masks_in: torch.Tensor,
        z: torch.Tensor,
        c: torch.Tensor,
        truncation_psi: float = 10,
        truncation_cutoff: Union[int, None] = None,
        skip_w_avg_update: bool = False,
        noise_mode: str = "none",
        return_stg1=False,
    ) -> torch.Tensor:
        """
        Forward function for `Generator`.

        Args:
            images_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            masks_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            z (torch.Tensor): Input tensor with shape (B, z_dim).
            c (torch.Tensor): Input tensor with shape (B, c_dim).
            truncation_psi (float, optional): Truncation psi. (default: 1)
            truncation_cutoff (Union[int, None], optional): Truncation cutoff. (default: None)
            skip_w_avg_update (bool, optional): Whether to skip w_avg update. (default: False)
            noise_mode (str, optional): Noise mode. (default: "none")
            return_stg1 (bool, optional): Whether to return the output of the first stage. (default: False) (unused)

        """
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update,
        )
        img = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        return img


class Discriminator(torch.nn.Module):
    """
    Class for the discriminator.


    Args:
        c_dim (int): Conditioning label (C) dimensionality.
        img_resolution (int): Input resolution.
        img_channels (int): Number of input color channels.
        channel_base (int, optional): Overall multiplier for the number of channels. (default: 32768) (unused)
        channel_max (int, optional): Maximum number of channels in any layer. (default: 512) (unused)
        channel_decay (int, optional): Channel decay. (default: 1) (unused)
        cmap_dim ([Union[int, None]], optional): Dimensionality of mapped conditioning label, None = default. (default: None)
        activation (str, optional): Activation function. (default: "lrelu")
        mbstd_group_size (int, optional): Group size for the minibatch standard deviation layer, None = entire minibatch. (default: 4)
        mbstd_num_channels (int, optional): Number of features for the minibatch standard deviation layer, 0 = disable. (default: 1)
    """

    def __init__(
        self,
        c_dim: int,
        img_resolution: int,
        img_channels: int,
        channel_base: int = 32768,
        channel_max: int = 512,
        channel_decay: int = 1,
        cmap_dim: Union[int, None] = None,
        activation: str = "lrelu",
        mbstd_group_size: int = 4,
        mbstd_num_channels: int = 1,
    ) -> None:
        """
        Constructor for `Discriminator`.

        Args:
            c_dim (int): Conditioning label (C) dimensionality.
            img_resolution (int): Input resolution.
            img_channels (int): Number of input color channels.
            channel_base (int, optional): Overall multiplier for the number of channels. (default: 32768) (unused)
            channel_max (int, optional): Maximum number of channels in any layer. (default: 512) (unused)
            channel_decay (int, optional): Channel decay. (default: 1) (unused)
            cmap_dim ([Union[int, None]], optional): Dimensionality of mapped conditioning label, None = default. (default: None)
            activation (str, optional): Activation function. (default: "lrelu")
            mbstd_group_size (int, optional): Group size for the minibatch standard deviation layer, None = entire minibatch. (default: 4)
            mbstd_num_channels (int, optional): Number of features for the minibatch standard deviation layer, 0 = disable. (default: 1)

        Returns:
            None
        """
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None
            )

        Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res - 1), activation))

        if mbstd_num_channels > 0:
            Dis.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis.append(
            Conv2dLayer(
                nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation
            )
        )
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2) * 4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

        # for 64x64
        Dis_stg1 = [DisFromRGB(img_channels + 1, nf(resolution_log2) // 2, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis_stg1.append(DisBlock(nf(res) // 2, nf(res - 1) // 2, activation))

        if mbstd_num_channels > 0:
            Dis_stg1.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis_stg1.append(
            Conv2dLayer(
                nf(2) // 2 + mbstd_num_channels,
                nf(2) // 2,
                kernel_size=3,
                activation=activation,
            )
        )
        self.Dis_stg1 = nn.Sequential(*Dis_stg1)

        self.fc0_stg1 = FullyConnectedLayer(
            nf(2) // 2 * 4**2, nf(2) // 2, activation=activation
        )
        self.fc1_stg1 = FullyConnectedLayer(
            nf(2) // 2, 1 if cmap_dim == 0 else cmap_dim
        )

    def forward(
        self,
        images_in: torch.Tensor,
        masks_in: torch.Tensor,
        images_stg1: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward function for `Discriminator`.

        Args:
            images_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            masks_in (torch.Tensor): Input tensor with shape (B, C, H, W).
            images_stg1 (torch.Tensor): Input tensor with shape (B, C, H, W).
            c (torch.Tensor): Input tensor with shape (B, c_dim).

        Returns:
            torch.Tensor: Output tensor with shape (B, 1) or (B, c_dim).

        """
        x = self.Dis(torch.cat([masks_in - 0.5, images_in], dim=1))
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        x_stg1 = self.Dis_stg1(torch.cat([masks_in - 0.5, images_stg1], dim=1))
        x_stg1 = self.fc1_stg1(self.fc0_stg1(x_stg1.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            x_stg1 = (x_stg1 * cmap).sum(dim=1, keepdim=True) * (
                1 / np.sqrt(self.cmap_dim)
            )

        return x, x_stg1


MAT_MODEL_URL = os.environ.get(
    "MAT_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth",
)


class MAT(InpaintModel):
    """
    MAT model for image inpainting.

    Args:
        device (torch.device): Device to load the model on.

    Inherited from `InpaintModel`.
    """

    min_size = 512
    pad_mod = 512
    pad_to_square = True

    def init_model(self, device: torch.device) -> None:
        """
        Construct the model and load the pretrained weights.

        Args:
            device (torch.device): Device to load the model on.

        Returns:
            None
        """
        seed = 240  # pick up a random number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3)
        self.model = load_model(G, MAT_MODEL_URL, device)
        self.z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)  # [1., 512]
        self.label = torch.zeros([1, self.model.c_dim], device=device)

    @staticmethod
    def is_downloaded() -> bool:
        """
        Check if the model is downloaded and implemented.

        Args:
            None

        Returns:
            bool: True if the model is downloaded and implemented.
        """
        return os.path.exists(get_cache_path_by_url(MAT_MODEL_URL))

    def forward(self, image, mask, config: Config):
        """
        Forward function for `MAT`.
        Input images and output images have same size

        Args:
            image (torch.Tensor): Input tensor with shape (H, W, C) RGB.
            mask (torch.Tensor): Input tensor with shape (H, W) => 0 or 255
            config (Config): Config object see schema.py

        Returns:
            torch.Tensor: Output tensor with shape (H, W, C) RGB.
        """

        image = norm_img(image)  # [0, 1]
        image = image * 2 - 1  # [0, 1] -> [-1, 1]

        mask = (mask > 127) * 255
        mask = 255 - mask
        mask = norm_img(mask)

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        output = self.model(
            image, mask, self.z, self.label, truncation_psi=1, noise_mode="none"
        )
        output = (
            (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        output = output[0].cpu().numpy()
        cur_res = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cur_res

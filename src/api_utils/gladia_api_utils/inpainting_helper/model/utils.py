import collections
import math
from itertools import repeat
from logging import getLogger
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
from torch import conv2d, conv_transpose2d

logger = getLogger(__name__)


def make_beta_schedule(
    device: torch.device,
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    """
    Make a beta schedule for the DDPM model.

    Args:
        device (torch.device): device to put the schedule on
        schedule (str): schedule type to use (linear, cosine, constant)
        n_timestep (int): number of timesteps to use for the schedule (number of training steps)
        linear_start (float): start of linear schedule (default: 1e-4)
        linear_end (float): end of linear schedule (default: 2e-2)
        cosine_s (float): cosine schedule parameter (default: 8e-3)

    Returns:
        np.ndarray: beta schedule
    """
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        ).to(device)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2).to(device)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas.numpy()


def make_ddim_sampling_parameters(
    alphacums: np.ndarray, ddim_timesteps: np.ndarray, eta: float
) -> Tuple:
    """
    Compute the parameters for the ddim sampler according to the DDIM formula provided in https://arxiv.org/abs/2010.02502

    Args:
        alphacums (np.ndarray): cumulative alphas for the DDIM model (shape: (num_timesteps, num_scales))
        ddim_timesteps (np.ndarray): timesteps to sample at (shape: (num_ddim_timesteps,))
        eta (float): eta parameter for the DDIM model

    Returns:
        Tuple: parameters for the ddim sampler (sigmas, alphas, alphas_prev)
    """
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )

    logger.debug(
        f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
    )
    logger.debug(
        f"For the chosen value of eta, which is {eta}, "
        f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
    )

    return sigmas, alphas, alphas_prev


def make_ddim_timesteps(
    ddim_discr_method: str, num_ddim_timesteps: int, num_ddpm_timesteps: int
) -> np.ndarray:
    """
    Make the timesteps for the DDIM sampler. These are the timesteps at which the DDIM sampler will be used. The DDIM
    sampler is used at the beginning of the training and then at regular intervals. The number of timesteps at which the
    DDIM sampler is used is controlled by the num_ddim_timesteps parameter.

    Args:
        ddim_discr_method (str): method to use for the DDIM sampler (uniform, quad)
        num_ddim_timesteps (int): number of timesteps at which the DDIM sampler is used
        num_ddpm_timesteps (int): number of timesteps for the DDPM model (number of training steps)

    Returns:
        np.ndarray: timesteps for the DDIM sampler
    """

    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1

    logger.debug(f"Selected timesteps for ddim sampler: {steps_out}")

    return steps_out


def noise_like(shape, device: torch.device, repeat: bool = False) -> torch.Tensor:
    """
    Generate noise with the same shape as the input. If repeat is True, the noise is repeated to match the shape of the
    input.

    Args:
        shape (tuple): shape of the input tensor (shape: (num_channels, height, width))
        device (torch.device): device to put the noise on (cpu or cuda)
        repeat (bool): whether to repeat the noise to match the shape of the input (default: False)

    Returns:
        torch.Tensor: noise tensor
    """

    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def timestep_embedding(
    device: torch.device,
    timesteps: np.ndarray,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings. The embeddings are created for the given timesteps and have the given
    dimension. The embeddings are created according to the formula provided in https://arxiv.org/abs/2010.02502.

    Args:
        device (torch.device): device to put the embeddings on (cpu or cuda)
        timesteps (np.ndarray): timesteps for which to create the embeddings (shape: (num_timesteps,)) it's a 1-D
        Tensor of N indices, one per batch element. These may be fractional.
        dim (int): dimension of the output's embeddings
        max_period (int): maximum period for the embeddings (default: 10000), controls the minimum frequency of the
        embeddings
        repeat_only (bool): whether to only repeat the embeddings to match the shape of the input (default: False) (unused)

    Returns:
        torch.Tensor: timestep of positional embeddings (shape: (num_timesteps, dim))
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=device)

    args = timesteps[:, None].float() * freqs[None]

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


###### MAT and FcF #######


def normalize_2nd_moment(
    x: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize the second moment of the input tensor. The second moment is computed along the given dimension. The second
    moment is normalized by dividing it by the mean of the second moment. The mean of the second moment is computed by
    taking the mean of the squared input tensor along the given dimension.

    Args:
        x (torch.Tensor): input tensor (shape: (batch_size, num_channels, height, width))
        dim (int): dimension along which to compute the second moment (default: 1)
        eps (float): epsilon to add to the mean of the second moment to avoid division by zero (default: 1e-8)

    Returns:
        torch.Tensor: normalized second moment (shape: (batch_size, num_channels, height, width))
    """

    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.

    Args:
        dict (dict): dictionary to convert to an EasyDict

    Returns:
        EasyDict: dictionary that allows access with the attribute syntax

    Example:
        >>> d = EasyDict({'a': 1, 'b': 2})
    """

    def __getattr__(self, name: str) -> Any:
        """
        Get the value of the given key. If the key is not in the dictionary, return None.

        Args:
            name (str): key to get the value for

        Returns:
            Any: value of the given key
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set the value of the given key.

        Args:
            name (str): key to set the value for
            value (Any): value to set for the given key

        Returns:
            None
        """
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """
        Delete the given key.

        Args:
            name (str): key to delete

        Returns:
            None
        """
        del self[name]


def _bias_act_ref(
    x: torch.Tensor,
    b: torch.Tensor = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Union[float, None] = None,
    gain: Union[float, None] = None,
    clamp: Union[float, None] = None,
) -> torch.Tensor:
    """
    Slow reference implementation of `bias_act()` using standard TensorFlow ops.

    Args:
        x (torch.Tensor): input tensor (shape: (batch_size, num_channels, height, width))
        b (torch.Tensor): bias tensor (shape: (num_channels,)) (default: None)
        dim (int): dimension along which to apply the bias (default: 1)
        act (str): activation function to apply (default: 'linear')
        alpha (Union[float, None]=None): alpha value for the leaky relu activation (default: None)
        gain (Union[float, None]=None): gain value for the leaky relu activation (default: None)
        clamp (Union[float, None]=None): clamp value for the leaky relu activation (default: None)

    Returns:
        torch.Tensor: output tensor (shape: (batch_size, num_channels, height, width))

    """
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


def bias_act(
    x: torch.Tensor,
    b: torch.Tensor = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Union[float, None] = None,
    gain: float = None,
    clamp: Union[float, None] = None,
):
    """
    Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x (torch.Tensor): input activation tensor Can be of any shape.
        b (torch.Tensor): Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                          as `x`. The shape must be known, and it must match the dimension of `x`
                          corresponding to `dim`. (default: None)
        dim (int): The dimension in `x` corresponding to the elements of `b`.
                   The value of `dim` is ignored if `b` is not specified. (default: 1)
        act (str): Name of the activation function to evaluate, or `"linear"` to disable.
                   (relu, lrelu, prelu, elu, selu, gelu, sigmoid, tanh, clamp, softplus, softsign, swish, mish) (default: 'linear')
                   See `activation_funcs` for a full list. `None` is not allowed.
        alpha (Union[float, None]=None): Shape parameter for the activation function (default: None)
        gain (Union[float, None]=None): Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1. (default: None)
        clamp (Union[float, None]=None): Clamp value for the activation function.
                                         Clamp the output values to `[-clamp, +clamp]`, or `None` to disable. (default: None)

    Returns:
        torch.Tensor: Output tensor of the same shape and type as `x`. If `b` is `None`, the output is `x`.
    """

    assert isinstance(x, torch.Tensor)

    return _bias_act_ref(
        x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp
    )


def _get_filter_size(filter: torch.Tensor) -> Tuple[int, int]:
    """
    Get the size of the filter.

    Args:
        filter (torch.Tensor): filter tensor (shape: (num_channels, num_filters, height, width))

    Returns:
        Tuple[int, int]: filter size (height, width)
    """

    if filter is None:
        return 1, 1

    assert isinstance(filter, torch.Tensor) and filter.ndim in [1, 2]
    filter_width = int(filter.shape[-1])
    filter_height = int(filter.shape[0])

    assert filter_width >= 1 and filter_height >= 1

    return filter_width, filter_height


def _get_weight_shape(
    weight: torch.Tensor,
) -> Tuple[int, int, int, int]:  # pylint: disable=too-many-branches
    """
    Get the shape of the weight tensor.

    Args:
        w (torch.Tensor): weight tensor (shape: (num_channels, num_filters, height, width))

    Returns:
        Tuple[int, int, int, int]: weight shape (num_channels, num_filters, height, width)
    """

    shape = [int(sz) for sz in weight.shape]

    return shape


def _parse_scaling(scaling: Union[int, list, tuple]) -> Tuple[int, int]:
    """
    Parse scaling. If scaling is a list or tuple, it must be of length 2. If it is an int, it is converted to a tuple.

    Args:
        scaling (Union[int, list, tuple]): scaling factor (height, width)

    Returns:
        Tuple[int, int]: scaling factor (height, width)
    """

    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1

    return sx, sy


def _parse_padding(padding: Union[int, list, tuple]) -> Tuple[int, int, int, int]:
    """
    Parse padding. If padding is a list or tuple, it must be of length 4. If it is an int, it is converted to a tuple. If it is a string, it is converted to a tuple.

    Args:
        padding (Union[int, list, tuple]): padding factor (left, right, top, bottom)

    Returns:
        Tuple[int, int, int, int]: padding factor (left, right, top, bottom)
    """

    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)

    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding

    return padx0, padx1, pady0, pady1


def setup_filter(
    filter: Union[torch.Tensor, np.ndarray, list, None],
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
    flip_filter: bool = False,
    gain: int = 1,
    return_separable: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        filter (Union[torch.Tensor, np.ndarray, list, None]): Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device (torch.device): device to place the filter on (default: torch.device("cpu"))
        normalize (bool): Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter (bool): Flip the filter? (default: False).
        gain (int): Scaling factor for the filter signal magnitude (default: 1).
        return_separable:   Return a separable filter? (default: True).

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Filter tensor of shape
                    Float32 tensor of the shape
                    `[filter_height, filter_width]` (non-separable) or
                    `[filter_taps]` (separable).
    """

    # Validate.
    if filter is None:
        filter = 1
    filter = torch.as_tensor(filter, dtype=torch.float32)
    assert filter.ndim in [0, 1, 2]
    assert filter.numel() > 0
    if filter.ndim == 0:
        filter = filter[np.newaxis]

    # Separable?
    if return_separable:
        return_separable = filter.ndim == 1 and filter.numel() >= 8
    if filter.ndim == 1 and not return_separable:
        filter = filter.ger(filter)
    assert filter.ndim == (1 if return_separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        filter /= filter.sum()
    if flip_filter:
        filter = filter.flip(list(range(filter.ndim)))
    filter = filter * (gain ** (filter.ndim / 2))
    filter = filter.to(device=device)

    return filter


def _ntuple(n: int) -> Callable[[int], Tuple[int, ...]]:
    """
    Generate a Callable function that Convert n to an n-tuple. If n is already an n-tuple, it is returned as-is.
    If n is a list or tuple, it is converted to a tuple.
    If n is an integer, it is converted to a tuple of length n, filled with n.

    Args:
        n (Union[int, list, tuple]): input

    Returns:
        Callable: function to convert a called x to an n-tuple
    """

    def parse(x: Union[collections.abc.Iterable, int]) -> Tuple[int, ...]:
        """
        Parse x to an n-tuple.

        Args:
            x (Union[collections.abc.Iterable, int]): input

        Returns:
            Tuple[int, ...]: n-tuple
        """

        if isinstance(x, collections.abc.Iterable):
            return x

        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

activation_funcs = {
    "linear": EasyDict(
        func=lambda x, **_: x,
        def_alpha=0,
        def_gain=1,
        cuda_idx=1,
        ref="",
        has_2nd_grad=False,
    ),
    "relu": EasyDict(
        func=lambda x, **_: torch.nn.functional.relu(x),
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=2,
        ref="y",
        has_2nd_grad=False,
    ),
    "lrelu": EasyDict(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2,
        def_gain=np.sqrt(2),
        cuda_idx=3,
        ref="y",
        has_2nd_grad=False,
    ),
    "tanh": EasyDict(
        func=lambda x, **_: torch.tanh(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=4,
        ref="y",
        has_2nd_grad=True,
    ),
    "sigmoid": EasyDict(
        func=lambda x, **_: torch.sigmoid(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=5,
        ref="y",
        has_2nd_grad=True,
    ),
    "elu": EasyDict(
        func=lambda x, **_: torch.nn.functional.elu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=6,
        ref="y",
        has_2nd_grad=True,
    ),
    "selu": EasyDict(
        func=lambda x, **_: torch.nn.functional.selu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=7,
        ref="y",
        has_2nd_grad=True,
    ),
    "softplus": EasyDict(
        func=lambda x, **_: torch.nn.functional.softplus(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=8,
        ref="y",
        has_2nd_grad=True,
    ),
    "swish": EasyDict(
        func=lambda x, **_: torch.sigmoid(x) * x,
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=9,
        ref="x",
        has_2nd_grad=True,
    ),
}


def upfirdn2d(
    x: torch.Tensor,
    f: torch.Tensor,
    up: Union[int, list, Tuple] = 1,
    down: Union[int, list, Tuple] = 1,
    padding: Union[int, list, Tuple] = 0,
    flip_filter: bool = False,
    gain: int = 1,
) -> torch.Tensor:
    """
    Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x (torch.Tensor): Float32/float64/float16 input tensor of the shape
                        `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape
                        `[filter_height, filter_width]` (non-separable),
                        `[filter_taps]` (separable), or
                        `None` (identity).
        up (Union[int, list, Tuple]): Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down (Union[int, list, Tuple]): Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding (Union[int, list, Tuple]): Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter (bool): False = convolution, True = correlation (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """

    return _upfirdn2d_ref(
        x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain
    )


def _upfirdn2d_ref(
    x: torch.Tensor,
    f: torch.Tensor,
    up: Union[int, list, Tuple] = 1,
    down: Union[int, list, Tuple] = 1,
    padding: Union[int, list, Tuple] = 0,
    flip_filter: bool = False,
    gain: int = 1,
) -> torch.Tensor:
    """
    Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.

    Args:
        x (torch.Tensor): Float32/float64/float16 input tensor of the shape
                        `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape
                        `[filter_height, filter_width]` (non-separable),
                        `[filter_taps]` (separable), or
                        `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a list/tuple
                        `[x, y]` (default: 1).
        down (int): Integer downsampling factor. Can be a single int or a list/tuple
                        `[x, y]` (default: 1).
        padding (int): Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        flip_filter (bool): False = convolution, True = correlation (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape

    upx, upy = up, up
    downx, downy = down, down

    padx0, padx1, pady0, pady1 = padding[0], padding[1], padding[2], padding[3]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(
        x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
    )
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def downsample2d(
    x: torch.Tensor,
    f: Union[torch.Tensor, None],
    down: Union[int, list, Tuple] = 2,
    padding: Union[int, list, Tuple] = 0,
    flip_filter: bool = False,
    gain: int = 1,
) -> torch.Tensor:
    """
    Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x (torch.Tensor): Float32/float64/float16 input tensor of the shape
                        `[batch_size, num_channels, in_height, in_width]`.
        f (Union[torch.Tensor, None]): Float32 FIR filter of the shape
                        `[filter_height, filter_width]` (non-separable),
                        `[filter_taps]` (separable), or
                        `None` (identity).
        down (Union[int, list, Tuple]): Integer downsampling factor. Can be a single int or a list/tuple
                        `[x, y]` (default: 2).
        padding (Union[int, list, Tuple]): Padding with respect to the upsampled image. Can be a single number or a list/tuple
                        `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        flip_filter (bool): False = convolution, True = correlation (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)

    padx0, padx1, pady0, pady1 = padding, padding, padding, padding

    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain)


def upsample2d(
    x: torch.Tensor,
    f: Union[torch.Tensor, None],
    up: Union[int, list, Tuple] = 2,
    padding: Union[int, list, Tuple] = 0,
    flip_filter: bool = False,
    gain: int = 1,
) -> torch.Tensor:
    """
    Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x (torch.Tensor): Float32/float64/float16 input tensor of the shape
                        `[batch_size, num_channels, in_height, in_width]`.
        f (Union[torch.Tensor, None]): Float32 FIR filter of the shape
                        `[filter_height, filter_width]` (non-separable),
                        `[filter_taps]` (separable), or
                        `None` (identity).
        up (Union[int, list, Tuple]): Integer upsampling factor. Can be a single int or a list/tuple
                        `[x, y]` (default: 2).
        padding (Union[int, list, Tuple]): Padding with respect to the downsampled image. Can be a single number or a list/tuple
                        `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        flip_filter (bool): False = convolution, True = correlation (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """

    upx, upy = _parse_scaling(up)

    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]

    return upfirdn2d(
        x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain * upx * upy
    )


class MinibatchStdLayer(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator. It calculates the standard deviation
    of the given input tensor over the minibatch dimension and appends it as a new feature map.
    Inheriting from torch.nn.Module is necessary for the layer to be recognized as a torch.nn.Module

    Args:
        group_size (int): Group size for computing the standard deviation.
        num_channels (int): Number of channels for the input tensor.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels + 1, height, width]`.
    """

    def __init__(self, group_size, num_channels=1):
        """
        Constructor method for the MinibatchStdLayer class.

        Args:
            group_size (int): Group size for computing the standard deviation.
            num_channels (int): Number of channels for the input tensor.

        Returns:
            None
        """

        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer

        Args:
            x (torch.Tensor): Input tensor of the shape [N, C, H, W] `[batch_size, num_channels, height, width]`.

        Returns:
            torch.Tensor: Tensor of the shape `[batch_size, num_channels + 1, height, width]`.
        """

        N, C, H, W = x.shape
        G = (
            torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
            if self.group_size is not None
            else N
        )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.

        return x


class FullyConnectedLayer(torch.nn.Module):
    """
    Fully connected layer for the discriminator. It calculates the standard deviation
    of the given input tensor over the minibatch dimension and appends it as a new feature map.
    Inheriting from torch.nn.Module is necessary for the layer to be recognized as a torch.nn.Module

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If set to True Apply additive bias before the activation function. (default: True)
        activation (Union[str, None]): Activation function to use. (relu, lrelu, tanh, sigmoid, None) (default: None)
        lr_multiplier (float): Learning rate multiplier for the weights. (default: 1.0)
        bias_init (float): Initial value for the additive bias. (default: 0.0)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "linear",
        lr_multiplier: float = 1,
        bias_init: float = 0,
    ):
        """
        Constructor method for the FullyConnectedLayer class.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If set to True Apply additive bias before the activation function. (default: True)
            activation (Union[str, None]): Activation function to use. (relu, lrelu, tanh, sigmoid, None) (default: None)
            lr_multiplier (float): Learning rate multiplier for the weights. (default: 1.0)
            bias_init (float): Initial value for the additive bias. (default: 0.0)

        Returns:
            None
        """

        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """

        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = x.matmul(w.t())
            out = x + b.reshape([-1 if i == x.ndim - 1 else 1 for i in range(x.ndim)])
        else:
            x = x.matmul(w.t())
            out = bias_act(x, b, act=self.activation, dim=x.ndim - 1)

        return out


def _conv2d_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    transpose: bool = False,
    flip_weight: bool = True,
):
    """
    Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.

    Args:
        x (torch.Tensor): Input tensor of shape [N, C, H, W].
                          N = batch size, C = number of channels, H = height, W = width.
                          Must be float32. Must be on the same device as `w`.
        w (torch.Tensor): Weight tensor of shape [C, K, R, S].
        stride (int): Convolution stride. Must be 1 for transposed convolutions. (default: 1)
        padding (int): Convolution padding. Must be 0 for transposed convolutions. (default: 0)
        groups (int): Number of convolution groups. (default: 1)
        transpose (bool): If True, use `conv_transpose2d()`. (default: False)
        flip_weight (bool): If True, flip the weight tensor along the spatial axes. (default: True)

    Returns:
        torch.Tensor: Output tensor of shape [N, K, H', W'].
    """
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)

    # Flip weight if requested.
    if (
        not flip_weight
    ):  # conv2d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if (
        kw == 1
        and kh == 1
        and stride == 1
        and padding in [0, [0, 0], (0, 0)]
        and not transpose
    ):
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape(
                    [in_shape[0], in_channels_per_group, -1]
                )
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = conv2d(x, w, groups=groups)
            return x.to(memory_format=torch.channels_last)

    # Otherwise => execute using conv2d_gradfix.
    op = conv_transpose2d if transpose else conv2d

    return op(x, w, stride=stride, padding=padding, groups=groups)


def conv2d_resample(
    x: torch.Tensor,
    w: torch.Tensor,
    f: Union[torch.Tensor, None] = None,
    up: int = 1,
    down: int = 1,
    padding: int = 0,
    groups: int = 1,
    flip_weight: bool = True,
    flip_filter: bool = False,
):
    """
    2D convolution with optional up/downsampling.
    Padding is performed only once at the beginning, not between the operations.

    Args:
        x (torch.Tensor): Input tensor of shape [N, C, H, W].
                          `[batch_size, in_channels, in_height, in_width]`.
                          Must be float32. Must be on the same device as `w`.
        w (torch.Tensor): Weight tensor of shape [C, K, R, S].
                          `[out_channels, in_channels//groups, kernel_height, kernel_width]`.
        f (torch.Tensor): Filter tensor of shape [1, 1, R', S'].
                          Low-pass filter for up/downsampling. Must be prepared beforehand by
                          calling setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype == x.dtype)
    assert f is None or (
        isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32
    )
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)

    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = _get_filter_size(f)

    px0, px1, py0, py1 = padding, padding, padding, padding

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d(
            x=x, f=f, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter
        )
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d(
            x=x,
            f=f,
            up=up,
            padding=[px0, px1, py0, py1],
            gain=up**2,
            flip_filter=flip_filter,
        )
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(
            x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight
        )
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(
                groups * in_channels_per_group, out_channels // groups, kh, kw
            )
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=[pyt, pxt],
            groups=groups,
            transpose=True,
            flip_weight=(not flip_weight),
        )
        x = upfirdn2d(
            x=x,
            f=f,
            padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
            gain=up**2,
            flip_filter=flip_filter,
        )
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(
                x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight
            )

    # Fallback: Generic reference implementation.
    x = upfirdn2d(
        x=x,
        f=(f if up > 1 else None),
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter,
    )
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x


class Conv2dLayer(torch.nn.Module):
    """
    2D convolution layer with optional up/downsampling and padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Width and height of the convolution kernel.
        bias (bool): Enable/disable bias.
        activation (str): Activation function: 'linear', 'relu', 'lrelu', 'tanh', 'sigmoid', 'softmax'. (default: 'linear')
        up (int): Upsampling factor. (default: 1)
        down (int): Downsampling factor. (default: 1)
        resample_kernel (list): Low-pass filter to apply when resampling activations. (default: [1,3,3,1])
        conv_clamp (Union[float, None]): Clamp the output of the convolution to +- this value.
                        Clamping is disabled when set to None (default: None)
        channels_last (bool): Use channels_last memory layout. (default: False)
        trainable (bool): Enable/disable weights updates during the training. (default: True)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
        activation: str = "linear",
        up: int = 1,
        down: int = 1,
        resample_filter: list = [
            1,
            3,
            3,
            1,
        ],
        conv_clamp: Union[float, None] = None,
        channels_last: bool = False,
        trainable: bool = True,
    ):
        """
        Constructor for the Conv2dLayer class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Width and height of the convolution kernel.
            bias (bool): Enable/disable bias.
            activation (str): Activation function: 'linear', 'relu', 'lrelu', 'tanh', 'sigmoid', 'softmax'. (default: 'linear')
            up (int): Upsampling factor. (default: 1)
            down (int): Downsampling factor. (default: 1)
            resample_kernel (list): Low-pass filter to apply when resampling activations. (default: [1,3,3,1])
            conv_clamp (Union[float, None]): Clamp the output of the convolution to +- this value.
                            Clamping is disabled when set to None (default: None)
            channels_last (bool): Use channels_last memory layout. (default: False)
            trainable (bool): Enable/disable weights updates during the training. (default: True)

        Returns:
            None
        """

        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            gain (float): Scaling factor. (default: 1.0)

        Returns:
            torch.Tensor: Output tensor.
        """

        w = self.weight * self.weight_gain
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )

        return out

import os
from logging import getLogger

import numpy as np
import torch

logger = getLogger(__name__)

from ..schema import Config, LDMSampler
from .base import InpaintModel
from .ddim_sampler import DDIMSampler
from .plms_sampler import PLMSSampler

torch.manual_seed(42)
from typing import Union

import torch.nn as nn

from ..helper import get_cache_path_by_url, load_jit_model, norm_img
from .utils import make_beta_schedule, timestep_embedding

LDM_ENCODE_MODEL_URL = os.environ.get(
    "LDM_ENCODE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_encode.pt",
)

LDM_DECODE_MODEL_URL = os.environ.get(
    "LDM_DECODE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_decode.pt",
)

LDM_DIFFUSION_MODEL_URL = os.environ.get(
    "LDM_DIFFUSION_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/diffusion.pt",
)


class DDPM(torch.nn.Module):
    """
    classic DDPM with Gaussian diffusion, in image space

    Args:
        device (torch.device): device to run the model on
        timesteps (int, optional): number of timesteps to run the model for. (default: 1000)
        beta_schedule (str, optional): type of beta schedule to use. (default: "linear")
        linear_start (float, optional): starting value for linear beta schedule. (default: 0.0015)
        linear_end (float, optional): ending value for linear beta schedule. (default: 0.0205)
        cosine_s (float, optional): s value for cosine beta schedule. (default: 0.008)
        original_elbo_weight (float, optional): weight for original elbo loss. (default: 0.0)
        v_posterior (float, optional): weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta. (default: 0.0)
        l_simple_weight (float, optional): weight for simple l loss. (default: 1.0)
        parameterization (str, optional): parameterization to use. (default: "eps")
        use_positional_encodings (bool, optional): whether to use positional encodings. (default: False)

    Inherited from torch.nn.Module:
    """

    def __init__(
        self,
        device: torch.device,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 0.0015,
        linear_end: float = 0.0205,
        cosine_s: float = 0.008,
        original_elbo_weight: float = 0.0,
        v_posterior: float = 0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight: float = 1.0,
        parameterization: str = "eps",  # all assuming fixed variance schedules
        use_positional_encodings: bool = False,
    ) -> None:
        """
        Constructor for DDPM class

        Args:
            device (torch.device): device to run the model on
            timesteps (int, optional): number of timesteps to run the model for. (default: 1000)
            beta_schedule (str, optional): type of beta schedule to use. (default: "linear")
            linear_start (float, optional): starting value for linear beta schedule. (default: 0.0015)
            linear_end (float, optional): ending value for linear beta schedule. (default: 0.0205)
            cosine_s (float, optional): s value for cosine beta schedule. (default: 0.008)
            original_elbo_weight (float, optional): weight for original elbo loss. (default: 0.0)
            v_posterior (float, optional): weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta. (default: 0.0)
            l_simple_weight (float, optional): weight for simple l loss. (default: 1.0)
            parameterization (str, optional): parameterization to use. (default: "eps")
            use_positional_encodings (bool, optional): whether to use positional encodings. (default: False)

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.parameterization = parameterization
        self.use_positional_encodings = use_positional_encodings

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.register_schedule(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ) -> None:
        """
        Register beta schedule

        Args:
            given_betas (list, optional): list of betas to use. (default: None)
            beta_schedule (str, optional): type of beta schedule to use. (default: "linear")
            timesteps (int, optional): number of timesteps to run the model for. (default: 1000)
            linear_start (float, optional): starting value for linear beta schedule. (default: 1e-4)
            linear_end (float, optional): ending value for linear beta schedule. (default: 2e-2)
            cosine_s (float, optional): s value for cosine beta schedule. (default: 8e-3)

        Returns:
            None
        """
        betas = make_beta_schedule(
            self.device,
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


class LatentDiffusion(DDPM):
    """
    Latent Diffusion Model

    Implementation of Latent Diffusion Models (https://arxiv.org/abs/2102.09672)
    In this model, the diffusion process is conditioned on a latent variable z that is sampled from a prior distribution p(z).
    The diffusion process is then conditioned on z and x_{t-1}, and the diffusion process is conditioned on the previous timestep x_{t-1}.

    Args:
        diffusion_model (DiffusionModel): diffusion model to use
        device (torch.device): device to use
        cond_stage_key (str, optional): key to use for conditional stage. (default: "image")
        cond_stage_trainable (bool, optional): whether to train the conditional stage. (default: False)
        concat_mode (bool, optional): whether to concatenate the latent variable z to the conditional stage. (default: True)
        scale_factor (float, optional): scale factor for the latent variable z. (default: 1.0)
        scale_by_std (bool, optional): whether to scale the latent variable z by the standard deviation of the prior distribution. (default: False)
        *args: positional arguments
        **kwargs: keyword arguments


    Inherits from DDPM
    """

    def __init__(
        self,
        diffusion_model,
        device,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ) -> None:
        """
        Constructor for LatentDiffusion

        Args:
            diffusion_model (DiffusionModel): diffusion model to use
            device (torch.device): device to use
            cond_stage_key (str, optional): key to use for conditional stage. (default: "image")
            cond_stage_trainable (bool, optional): whether to train the conditional stage. (default: False)
            concat_mode (bool, optional): whether to concatenate the latent variable z to the conditional stage. (default: True)
            scale_factor (float, optional): scale factor for the latent variable z. (default: 1.0)
            scale_by_std (bool, optional): whether to scale the latent variable z by the standard deviation of the prior distribution. (default: False)
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            None
        """

        self.num_timesteps_cond = 1
        self.scale_by_std = scale_by_std
        super().__init__(device, *args, **kwargs)
        self.diffusion_model = diffusion_model
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 2
        self.scale_factor = scale_factor

    def make_cond_schedule(
        self,
    ) -> None:
        """
        Make conditional schedule for the diffusion process

        Args:
            None

        Returns:
            None
        """
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def register_schedule(
        self,
        given_betas: Union[torch.Tensor, None] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Register schedule for the diffusion process

        Args:
            given_betas (Union[torch.Tensor, None]): given betas to use. (default: None)
            beta_schedule (str, optional): beta schedule to use. (default: "linear")
            timesteps (int, optional): number of timesteps to use. (default: 1000)
            linear_start (float, optional): starting beta for linear schedule. (default: 1e-4)
            linear_end (float, optional): ending beta for linear schedule. (default: 2e-2)
            cosine_s (float, optional): s parameter for cosine schedule. (default: 8e-3)

        Returns:
            None
        """
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def apply_model(
        self, x_noisy: torch.Tensor, t: int, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the diffusion model to the input x_noisy at timestep t.

        Args:
            x_noisy (torch.Tensor): input tensor
            t (int): timestep
            cond (torch.Tensor): conditioning tensor

        Returns:
            torch.Tensor: output tensor
        """

        t_emb = timestep_embedding(x_noisy.device, t, 256, repeat_only=False)
        x_recon = self.diffusion_model(x_noisy, t_emb, cond)
        return x_recon


class LDM(InpaintModel):
    """
    Class for Latent Diffusion Model

    Args:
        device (torch.device): The device to run the model on.
        fp16 (bool): Whether to use mixed precision training. (default: True)

    Inherit from InpaintModel

    """

    pad_mod = 32

    def __init__(self, device: torch.device, fp16: bool = True) -> None:
        """
        Constructor for the LDM model.

        Args:
            device (torch.device): The device to run the model on.
            fp16 (bool): Whether to use mixed precision training. (default: True)

        Returns:
            None
        """
        self.fp16 = fp16
        super().__init__(device)
        self.device = device

    def init_model(self, device) -> None:
        """
        Initialize the model.

        Args:
            device (torch.device): device to run the model on

        Returns:
            None
        """
        self.diffusion_model = load_jit_model(LDM_DIFFUSION_MODEL_URL, device)
        self.cond_stage_model_decode = load_jit_model(LDM_DECODE_MODEL_URL, device)
        self.cond_stage_model_encode = load_jit_model(LDM_ENCODE_MODEL_URL, device)
        if self.fp16 and "cuda" in str(device):
            self.diffusion_model = self.diffusion_model.half()
            self.cond_stage_model_decode = self.cond_stage_model_decode.half()
            self.cond_stage_model_encode = self.cond_stage_model_encode.half()

        self.model = LatentDiffusion(self.diffusion_model, device)

    @staticmethod
    def is_downloaded() -> bool:
        """
        Check if the model is implemented and downloaded

        Returns:
            bool: True if the model is implemented and downloaded
        """
        model_paths = [
            get_cache_path_by_url(LDM_DIFFUSION_MODEL_URL),
            get_cache_path_by_url(LDM_DECODE_MODEL_URL),
            get_cache_path_by_url(LDM_ENCODE_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    @torch.cuda.amp.autocast()
    def forward(self, image, mask, config: Config):
        """
        Forward pass of the model. Returns the reconstructed image.

        Args:
            image (torch.Tensor): The input image. Shape[H, W, C] RGB float32 (ex: [1,3,512,512])
            mask (torch.Tensor): The mask. Shape[H, W, 1] float32 (ex: [1, 1, 512, 512])
            config (Config): The config. See Schema.py for more details.

        Returns:
            torch.Tensor: The reconstructed image BGR
        """
        # image [1,3,512,512] float32
        # mask: [1,1,512,512] float32
        # masked_image: [1,3,512,512] float32
        if config.ldm_sampler == LDMSampler.ddim:
            sampler = DDIMSampler(self.model)
        elif config.ldm_sampler == LDMSampler.plms:
            sampler = PLMSSampler(self.model)
        else:
            raise ValueError()

        steps = config.ldm_steps
        image = norm_img(image)
        mask = norm_img(mask)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        masked_image = (1 - mask) * image

        mask = self._norm(mask)
        masked_image = self._norm(masked_image)

        c = self.cond_stage_model_encode(masked_image)
        torch.cuda.empty_cache()

        cc = torch.nn.functional.interpolate(mask, size=c.shape[-2:])  # 1,1,128,128
        c = torch.cat((c, cc), dim=1)  # 1,4,128,128

        shape = (c.shape[1] - 1,) + c.shape[2:]
        samples_ddim = sampler.sample(
            steps=steps, conditioning=c, batch_size=c.shape[0], shape=shape
        )
        torch.cuda.empty_cache()
        x_samples_ddim = self.cond_stage_model_decode(
            samples_ddim
        )  # samples_ddim: 1, 3, 128, 128 float32
        torch.cuda.empty_cache()

        inpainted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        inpainted_image = inpainted_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        inpainted_image = inpainted_image.astype(np.uint8)[:, :, ::-1]

        return inpainted_image

    def _norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize image tensor

        Args:
            tensor (torch.Tensor): Image tensor

        Returns:
            torch.Tensor: Normalized image tensor
        """

        return tensor * 2.0 - 1.0

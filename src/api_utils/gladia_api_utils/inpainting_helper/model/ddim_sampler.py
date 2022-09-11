from logging import getLogger
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from .utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

logger = getLogger(__name__)


class DDIMSampler(object):
    """
    Class for sampling from the DDIM model.

    Args:

    """

    def __init__(self, model, schedule: str = "linear") -> None:
        """
        Constructor for DDIMSampler.

        Args:
            model (DDIM): DDIM model to sample from.
            schedule (str): Schedule for sampling. Either "linear" or "exponential". (default: "linear")

        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name: str, attr: torch.Tensor) -> None:
        """
        Register a buffer.

        Args:
            name (str): Name of the buffer.
            attr (torch.Tensor): Attribute to register.

        Returns:
            None
        """
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
    ):
        """
        Make the schedule for sampling.

        Args:
            ddim_num_steps (int): Number of steps to sample.
            ddim_discretize (str): Discretization method. Either "uniform" or "log". (default: "uniform")
            ddim_eta (float): Eta parameter for the log discretization. (default: 0.0)

        Returns:
            None
        """
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
        )
        alphas_cumprod = self.model.alphas_cumprod  # torch.Size([1000])
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        steps: int,
        conditioning: torch.Tensor,
        batch_size: int = 1,
        shape: Tuple[int, int, int] = None,
    ) -> torch.Tensor:
        """
        Samples from the DDIM model.

        Args:
            steps (int): Number of steps to sample.
            conditioning (torch.Tensor): Conditioning tensor.
            batch_size (int): Batch size.
            shape (Tuple[int, int, int]): Shape of the image.

        Returns:
            torch.Tensor: Samples from the DDIM model.
        """
        self.make_schedule(ddim_num_steps=steps, ddim_eta=0)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # samples: 1,3,128,128
        return self.ddim_sampling(
            conditioning,
            size,
            quantize_denoised=False,
            ddim_use_original_steps=False,
            noise_dropout=0,
            temperature=1.0,
        )

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, int, int],
        ddim_use_original_steps: int = False,
        quantize_denoised: int = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Sample from the DDIM model.

        Args:
            cond (torch.Tensor): Conditioning tensor.
            shape (Tuple[int, int, int]): Shape of the sample.
            ddim_use_original_steps (bool): Whether to use the original DDPM sampling steps. (default: False)
            quantize_denoised (bool): Whether to quantize the denoised image. (default: False)
            temperature (float): Temperature for sampling. (default: 1.0)
            noise_dropout (float): Dropout rate for noise. (default: 0.0)

        Returns:
            torch.Tensor: Sampled image.
        """
        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device, dtype=cond.dtype)
        timesteps = (
            self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        )

        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
            )
            img, _ = outs

        return img

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the diffusion process at a given timestep.

        Args:
            x (torch.Tensor): the input image
            c (torch.Tensor): the conditioning
            t (torch.Tensor): the timestep
            index (int): the index of the timestep
            repeat_noise (bool): whether to repeat the noise for each timestep (default: False)
            use_original_steps (bool): whether to use the original number of timesteps (default: False)
            quantize_denoised (bool): whether to quantize the denoised image (default: False)
            temperature (float): the temperature for sampling (default: 1.0)
            noise_dropout (float): the dropout rate for the noise (default: 0.0)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: previous image, denoised image
        """
        b, *_, device = *x.shape, x.device
        e_t = self.model.apply_model(x, t, c)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:  # 没用
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:  # 没用
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

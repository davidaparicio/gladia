# From: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/plms.py
from logging import getLogger
from typing import Callable, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

logger = getLogger(__name__)


class PLMSSampler(object):
    """
    PLMS sampling class.

    Args:
        model (nn.Module): model to sample from
        schedule (str): schedule to use for sampling (linear, geometric, geometric2) (default: linear)

    Returns:
        PLMSSampler: sampler object
    """

    def __init__(self, model, schedule="linear", **kwargs):
        """
        Constructor method for PLMSSampler class.

        Args:
            model (nn.Module): model to sample from
            schedule (str): schedule to use for sampling (linear, geometric, geometric2) (default: linear)

        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, buffer_name: str, attr_to_register: torch.Tensor) -> None:
        """
        Register buffer. This is a wrapper for torch.nn.Module.register_buffer.

        Args:
            buffer_name (str): name of the buffer
            attr_to_register (torch.Tensor): attribute to register
        """

        setattr(self, buffer_name, attr_to_register)

    def make_schedule(
        self,
        ddim_num_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
    ) -> None:
        """
        Make schedule for PLMS sampling.

        Args:
            ddim_num_steps (int): number of steps
            ddim_discretize (str): discretization method (default: uniform)
            ddim_eta (float): eta parameter for geometric schedule (default: 0.0)

        Returns:
            None
        """

        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
        )
        alphas_cumprod = self.model.alphas_cumprod
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
        batch_size: int,
        shape: Tuple[int, int, int],
        conditioning: Union[torch.Tensor, None] = None,
        callback: Union[Callable, None] = None,
        img_callback: Union[Callable, None] = None,
        quantize_x0: bool = False,
        eta: float = 0.0,
        mask: Union[torch.Tensor, None] = None,
        x0: Union[torch.Tensor, None] = None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Union[Callable, None] = None,
        corrector_kwargs: Union[dict, None] = None,
        x_T: Union[torch.Tensor, None] = None,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Union[torch.Tensor, None] = None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from the model. This is a wrapper for the sample_from_model function.

        Args:
            steps (int): number of steps to sample
            batch_size (int): batch size
            shape (tuple): shape of the image
            conditioning (Union[torch.Tensor, None]): conditioning for the model (default: None)
            callback (Union[Callable, None]): callback function (default: None)
            img_callback (Union[Callable, None]): image callback function (default: None)
            quantize_x0 (bool): whether to quantize x0 (default: False)
            eta (float): eta parameter for geometric schedule (default: 0.0)
            mask (Union[torch.Tensor, None]): mask for the model (default: None)
            x0 (Union[torch.Tensor, None]): initial image (default: None)
            temperature (float): temperature for sampling (default: 1.0)
            noise_dropout (float): dropout rate for noise (default: 0.0)
            score_corrector (Union[Callable, None]): score corrector function (default: None)
            corrector_kwargs (Union[dict, None]): kwargs for score corrector (default: None)
            x_T (Union[torch.Tensor, None]): target image (default: None)
            log_every_t (int): log every t steps (default: 100)
            unconditional_guidance_scale (float): scale for unconditional guidance (default: 1.0)
            unconditional_conditioning (Union[torch.Tensor, None]): unconditional conditioning (default: None)

        Returns:
            torch.Tensor: samples
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    logger.debug(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    logger.debug(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        logger.debug(f"Data shape for PLMS sampling is {size}")

        samples = self.plms_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )

        return samples

    @torch.no_grad()
    def plms_sampling(
        self,
        cond: Union[torch.Tensor, None] = None,
        shape: Tuple = (1, 3, 256, 256),
        x_T: Union[torch.Tensor, None] = None,
        ddim_use_original_steps: bool = False,
        callback=None,
        timesteps=None,
        quantize_denoised: bool = False,
        mask=None,
        x0=None,
        img_callback=None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Union[Callable, None] = None,
        corrector_kwargs: Union[dict, None] = None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Sample from the model. This is a wrapper for the sample_from_model function.

        Args:
            cond (Union[torch.Tensor, None]): conditioning for the model (default: None)
            shape (tuple): shape of the image
            x_T (Union[torch.Tensor, None]): target image (default: None)
            ddim_use_original_steps (bool): whether to use original steps (default: False)
            callback (Union[Callable, None]): callback function (default: None)
            timesteps (Union[int, None]): number of timesteps (default: None)
            quantize_denoised (bool): whether to quantize x0 (default: False)
            mask (Union[torch.Tensor, None]): mask for the model (default: None)
            x0 (Union[torch.Tensor, None]): initial image (default: None)
            img_callback (Union[Callable, None]): image callback function (default: None)
            temperature (float): temperature for sampling (default: 1.0)
            noise_dropout (float): dropout rate for noise (default: 0.0)
            score_corrector (Union[Callable, None]): score corrector function (default: None)
            corrector_kwargs (Union[dict, None]): kwargs for score corrector (default: None)
            unconditional_guidance_scale (float): scale for unconditional guidance (default: 1.0)
            unconditional_conditioning (Union[torch.Tensor, None]): unconditional conditioning (default: None)

        Returns:
            torch.Tensor: samples
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = (
            list(reversed(range(0, timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        logger.debug(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="PLMS Sampler", total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_plms(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

        return img

    @torch.no_grad()
    def p_sample_plms(
        self,
        x: torch.Tensor,
        c: Union[torch.Tensor, None],
        t: torch.Tensor,
        index: int,
        repeat_noise: bool = False,
        use_original_steps: bool = False,
        quantize_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Union[Callable, None] = None,
        corrector_kwargs: Union[dict, None] = None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Union[torch.Tensor, None] = None,
        old_eps: Union[list, None] = None,
        t_next: Union[torch.Tensor, None] = None,
    ):
        """
        Sample from the model. This is a wrapper for the sample_from_model function. This is the
        PLMS version of the sampling function.

        Args:
            x (torch.Tensor): input image
            c (Union[torch.Tensor, None]): conditioning for the model (default: None)
            t (torch.Tensor): timestep
            index (int): index of the timestep
            repeat_noise (bool): whether to repeat noise (default: False)
            use_original_steps (bool): whether to use original steps (default: False)
            quantize_denoised (bool): whether to quantize x0 (default: False)
            temperature (float): temperature for sampling (default: 1.0)
            noise_dropout (float): dropout rate for noise (default: 0.0)
            score_corrector (Union[Callable, None]): score corrector function (default: None)
            corrector_kwargs (Union[dict, None]): kwargs for score corrector (default: None)
            unconditional_guidance_scale (float): scale for unconditional guidance (default: 1.0)
            unconditional_conditioning (Union[torch.Tensor, None]): unconditional conditioning (default: None)
            old_eps (Union[list, None]): list of old epsilons (default: None)
            t_next (Union[torch.Tensor, None]): next timestep (default: None)

        Returns:
            torch.Tensor: samples
        """
        b, *_, device = *x.shape, x.device

        def get_model_output(
            x: torch.Tensor, t: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get the model output for the given timestep and image.

            Args:
                x (torch.Tensor): Image tensor of shape (b, c, h, w) to sample from.
                t (torch.Tensor): Timestep tensor of shape (b,) to sample from.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (x, pred_x0, e_t)
            """
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            return e_t

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

        def get_x_prev_and_pred_x0(
            e_t: torch.Tensor, index: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get x_prev and pred_x0 for the current timestep index

            Args:
                e_t (torch.Tensor): The score of the current timestep (b, c, h, w)
                index (int): The current timestep index (0 is the last timestep)

            Returns:
                x_prev (Tuple[torch.Tensor, torch.Tensor]): The previous timestep image (b, c, h, w), and the previous timestep score (b, c, h, w)
            """
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature

            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

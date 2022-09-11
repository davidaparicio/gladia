import torch

from .model.fcf import FcF
from .model.lama import LaMa
from .model.ldm import LDM
from .model.mat import MAT
from .model.zits import ZITS
from .schema import Config

models = {"lama": LaMa, "ldm": LDM, "zits": ZITS, "mat": MAT, "fcf": FcF}


class ModelManager:
    """
    Inpainting model manager

    Args:
        name (str): model name (lama, ldm, zits, mat, fcf)
        device (torch.device): device to run the model on (cpu, cuda)

    Returns:
        ModelManager: the model manager
    """

    def __init__(self, name: str) -> None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(False)
        except Exception:
            pass

        self.name = name
        self.device = device
        self.model = self.init_model(name, device)

    def init_model(self, name: str, device) -> torch.nn.Module:
        """
        Initialize the model based on it's name

        Args:
            name (str): model name
            device (torch.device): device to run the model on

        Returns:
            torch.nn.Module: the model
        """
        if name in models:
            model = models[name](device)
        else:
            raise NotImplementedError(f"Not supported model: {name}")
        return model

    def is_downloaded(self, name: str) -> bool:
        """
        Check if the model is implemented valid and download it if necessary.

        Args:
            name (str): model name

        Returns:
            bool: True if the model is valid and downloaded, False otherwise
        """

        if name in models:
            return models[name].is_downloaded()
        else:
            raise NotImplementedError(f"Not supported model: {name}")

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor, config: Config
    ) -> torch.Tensor:
        """
        Make the class callable as a function

        Args:
            image (torch.Tensor): image tensor
            mask (torch.Tensor): mask tensor
            config (Config): model configuration (see schema.py)
        """
        return self.model(image, mask, config)

    def switch(self, new_name: str) -> None:
        """
        Switch the model to a new one

        Args:
            new_name (str): the new model name to initialize

        Returns:
            None
        """

        if new_name == self.name:
            return
        try:
            self.model = self.init_model(new_name, self.device)
            self.name = new_name
        except NotImplementedError as e:
            raise e

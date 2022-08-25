# adapted from https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing#scrollTo=BPnyd-XUKbfE
import torch
import torchvision
from gladia_api_utils.io import _open
from gladia_api_utils.model_management import download_models
from numpy import asarray
from PIL import Image


def predict(image: Image, steps: int = 10) -> Image:
    # Adapted from https://github.com/CompVis/latent-diffusion/blob/main/notebook_helpers.py

    from einops import rearrange, repeat
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config, ismap
    from omegaconf import OmegaConf

    ckpt_url = {
        "checkpoints": {
            "url": "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1",
            "output_path": "last.ckpt",
        },
    }
    conf_url = {
        "config": {
            "url": "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1",
            "output_path": "project.yaml",
        }
    }

    ckpt_path = download_models(ckpt_url)
    conf_path = download_models(conf_url)

    path_conf = conf_path["config"]["output_path"]
    path_ckpt = ckpt_path["checkpoints"]["output_path"]

    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)

    logs = run(model["model"], _open(image), "superresolution", custom_steps=steps)

    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1.0, 1.0)
    sample = (sample + 1.0) / 2.0 * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))

    return Image.fromarray(sample[0])
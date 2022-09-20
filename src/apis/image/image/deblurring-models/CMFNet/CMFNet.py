from PIL import Image
from skimage import img_as_ubyte
from collections import OrderedDict

from torch import device as torch_device
from torch import load as torch_load
from torch import clamp as torch_clamp
from torch import no_grad as torch_no_grad
from torch.cuda import is_available as is_cuda_available

from torch.nn import Module
from torch.nn.functional import pad
from torchvision.transforms.functional import to_tensor

from gladia_api_utils.io import _open
from gladia_api_utils.model_management import download_model
from apis.image.image.deblurring_models.CMFNet.model.CMFNet import CMFNet


MODEL_PATH = download_model(
    url="https://github.com/FanChiMao/CMFNet/releases/download/v0.0/deblur_GoPro_CMFNet.pth",
    output_path="deblur_GoPro_CMFNet.pth",
    uncompress_after_download=False,
)


def load_checkpoint(model: Module, weights: str) -> None:
    """
    Loads a checkpoint into a model

    Args:
        model (torch.nn.Module): Model to load weights to.
        weights (str): Path to checkpoint.

    Returns:
        None
    """
    checkpoint = torch_load(weights, map_location=torch_device("cpu"))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)



def predict(image: bytes) -> Image:
    """
    deblurs an image using CMFNet

    Args:
        image (bytes): Image to deblur.

    Returns:
        Image: Deblurred image.
    """

    image = _open(image).convert("RGB")

    basewidth = 512
    wpercent = basewidth / float(image.size[0])
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize), Image.BILINEAR)

    model = CMFNet()
    device = torch_device("cuda" if is_cuda_available() else "cpu")
    model = model.to(device)
    model.eval()
    load_checkpoint(model, MODEL_PATH)

    MUL = 8
    input_ = to_tensor(image).unsqueeze(0).to(device)

    # Pad the input if not_multiple_of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + MUL) // MUL) * MUL, ((w + MUL) // MUL) * MUL
    padh = H - h if h % MUL != 0 else 0
    padw = W - w if w % MUL != 0 else 0
    input_ = pad(input_, (0, padw, 0, padh), "reflect")

    with torch_no_grad():
        restored = model(input_)

    restored = torch_clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    return Image.fromarray(restored)

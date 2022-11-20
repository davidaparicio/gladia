import importlib.util
import json
import os
import sys
import urllib.parse
from logging import getLogger

from PIL import Image

logger = getLogger(__name__)

HELP_STRING = """
python <PATH_TO_FILE>/run_process.py <module_path> <model> <output_tmp_result> <kwargs for predict>
    - module_path : the route to the targeted model (for instance `apis/image/image/face-blurings/ageitgey/`)
    - model : the targeted model name (for instance `ageitgey`)
    - output_tmp_result : the path to where the results will be written (for example `/tmp/tmpo8q0coqe`)
    - kwargs : a dictionary encoded with `urllib.parse.quote()`. Keywords argument are needed for the `predict` function (for instance `{'image': '/tmp/tmp342by415'}` is encoded into `%257B%2522image%2522%253A%2520%2522/tmp/tmp342by415%2522%257D`)
"""

if __name__ == "__main__":

    if len(sys.argv) < 5:
        logger.fatal("Not enough arguments. Please read usage below.", HELP_STRING)

        sys.exit(1)

    module_path = sys.argv[1]
    model = sys.argv[2]
    output_tmp_result = sys.argv[3]
    kwargs = json.loads(urllib.parse.unquote(sys.argv[4]))

    PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app")

    os.environ[
        "LD_LIBRARY_PATH"
    ] = "/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/conda/lib"

    # if module_path is not absolute
    # then prepend the PATH_TO_GLADIA_SRC
    if not os.path.isabs(module_path):
        module_path = os.path.join(PATH_TO_GLADIA_SRC, module_path)

    spec = importlib.util.spec_from_file_location(
        module_path,
        os.path.join(module_path, f"{model}.py"),
    )

    sys.path.append(module_path)

    this_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(this_module)

    output = this_module.predict(**kwargs)

    if isinstance(output, Image.Image):
        output.save(f"{output_tmp_result}", format="PNG")
    elif isinstance(output, bytes):
        with open(f"{output_tmp_result}", "wb") as f:
            f.write(output)
    else:
        json.dump(output, open(output_tmp_result, "w"))

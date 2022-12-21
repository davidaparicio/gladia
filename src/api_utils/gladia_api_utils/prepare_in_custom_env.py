import importlib.util
import os
import sys
from logging import getLogger

logger = getLogger(__name__)

GLADIA_PERSISTENT_PATH = os.getenv("GLADIA_PERSISTENT_PATH", "/gladia")
MAMBA_ROOT_PREFIX = os.getenv("MAMBA_ROOT_PREFIX", f"{GLADIA_PERSISTENT_PATH}/conda")

HELP_STRING = """
python <PATH_TO_FILE>/prepare_in_custom_env.py <module_path> <model>
    - module_path : the route to the targeted model (for instance `apis/image/image/face-blurings/ageitgey/`)
    - model : the targeted model name (for instance `ageitgey`)
"""

if __name__ == "__main__":

    if len(sys.argv) < 3:
        logger.critical("Not enough arguments. Please read usage below.", HELP_STRING)

        sys.exit(1)

    module_path = sys.argv[1]
    model = sys.argv[2]

    PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app")

    os.environ[
        "LD_LIBRARY_PATH"
    ] = f"/usr/local/nvidia/lib64:/usr/local/cuda/lib64:{MAMBA_ROOT_PREFIX}/lib"

    # if module_path is not absolute
    # then prepend the PATH_TO_GLADIA_SRC
    if not os.path.isabs(module_path):
        module_path = os.path.join(PATH_TO_GLADIA_SRC, module_path)

    spec = importlib.util.spec_from_file_location(
        module_path,
        os.path.join(module_path, f"{model}.py"),
    )

    sys.path.append(module_path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if hasattr(module, "prepare"):
        module.prepare()

    sys.exit(0)

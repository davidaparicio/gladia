import distutils.dir_util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from posixpath import isabs
from typing import Union

from genericpath import isdir
from torch.cuda import is_available as cuda_is_available

GLADIA_SRC_PATH = os.getenv("GLADIA_SRC_PATH", "/app")


def load_json(path: str) -> dict:
    """
    Load a json file.

    Args:
        path (str): path to json file

    Returns:
        dict: json file as a dictionary
    """

    with open(path, "r") as f:
        data = json.load(f)

    return data


def load_config() -> dict:
    """Load the config.json file from the GLADIA_SRC_PATH.

    Returns:
        The config.json file as a dictionary.
    """
    return load_json(os.path.join(GLADIA_SRC_PATH, "config.json"))


def load_models_config() -> dict:
    """
    Load the models config from the GLADIA_SRC_PATH.
    Args:
        None

    Returns:
        dict: spacy config
    """
    return load_json(os.path.join(GLADIA_SRC_PATH, "models-config.json"))


def copy(source: str, destination: str) -> None:
    """
    Copy file from source to destination.

    Args:
        source (str): source path to copy from
        destination (str): destination path to copy to

    Returns:
        None
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not isabs(Path(source)):
        source = os.path.join(root_path, source)
    if not isabs(Path(destination)):
        destination = os.path.join(root_path, destination)

    distutils.dir_util.copy_tree(source, destination)


def get_cwd():
    """
    Get the current working directory.

    Returns:
        str: current working directory
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    return root_path


def path_to_absolute(path: str) -> str:
    """
    Make a path absolute if it's not already.

    Args:
        path (str): path to make absolute

    Returns:
        str: absolute path
    """

    if not isabs(Path(path)):
        # used for relative paths
        namespace = sys._getframe(1).f_globals
        cwd = os.getcwd()
        rel_path = namespace["__file__"]
        root_path = os.path.dirname(os.path.join(cwd, rel_path))
        path = os.path.join(root_path, path)

    return path


def run(*argv) -> subprocess.CompletedProcess:
    """
    Run a command on the system.

    Args:
        *argv (str): command to run

    Returns:
        str: output of command
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    cmd = f"cd {root_path} &&"
    for arg in argv:
        cmd += f" {arg}"

    return subprocess.run([cmd], shell=True, capture_output=True)


def remove(*paths) -> None:
    """
    Remove a file or directory.

    Args:
        *paths (str): path to remove

    Returns:
        None
    """

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
            if not isabs(path):
                namespace = sys._getframe(1).f_globals
                cwd = os.getcwd()
                rel_path = namespace["__file__"]
                root_path = os.path.dirname(os.path.join(cwd, rel_path))
                path = os.path.join(root_path, path)
            if isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def get_first_available_gpu_id() -> int:
    """
    Get the first available GPU id. If no GPUs are available, return None.

    Returns:
        int: first available GPU id return None if no GPU is available
    """

    available_gpu_ids = get_available_gpu_ids()

    return None if len(available_gpu_ids) == 0 else available_gpu_ids[0]


def get_random_available_gpu_id() -> Union[int, None]:
    """
    Get a random available GPU id. If no GPUs are available, return None.

    Returns:
        int: random available GPU id return None if no GPU is available
    """

    available_gpu_ids = get_available_gpu_ids()

    if available_gpu_ids:
        return random.choice(available_gpu_ids)

    return None


def get_available_gpu_ids() -> list:
    """
    Get all available GPU ids. If no GPUs are available, return an empty list.

    Returns:
        list: all available GPU ids return an empty list if no GPU is available
    """

    gpu_ids = list()

    if cuda_is_available() and (
        cuda_visible_devices := os.getenv("CUDA_VISIBLE_DEVICES", None)
    ):
        gpu_ids = [int(x) for x in cuda_visible_devices.split(",")]

    return gpu_ids

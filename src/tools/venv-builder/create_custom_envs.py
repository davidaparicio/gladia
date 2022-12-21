import argparse
import filecmp
import logging
import os
import re
import shutil
import subprocess
import tempfile
from logging import getLogger
from typing import List, Tuple

import yaml
from gladia_api_utils import get_activated_task_path
from tqdm import tqdm

logger = getLogger(__name__)


ENV_DEFAULT_FILENAME = "env.yaml"

PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app/src")
GLADIA_PERSISTENT_PATH = os.getenv("GLADIA_PERSISTENT_PATH", "/gladia")
MAMBA_ROOT_PREFIX = os.getenv("MAMBA_ROOT_PREFIX", f"{GLADIA_PERSISTENT_PATH}/conda")


def retrieve_package_from_env_file(env_file: dict) -> Tuple[List[str], List[str]]:
    """
    retrieve the necessary packages to install from the env file

    Args:
        env_file (dict): env file to use to retrieve the packages to install from

    Returns:
        Tuple[List[str], List[str]]: Tuple of the packages to install from pip and from channel

    Raises:
        RuntimeError: If the env file is empty
    """
    packages_to_install_from_pip = []
    packages_to_install_from_channel = []

    if env_file is None or "dependencies" not in env_file.keys():
        return packages_to_install_from_pip, packages_to_install_from_channel

    for package in env_file["dependencies"]:
        if type(package) == dict and "pip" in package.keys():
            for pip_package in package["pip"]:
                packages_to_install_from_pip.append(pip_package)
        else:
            packages_to_install_from_channel.append(package)

    return packages_to_install_from_pip, packages_to_install_from_channel


def create_temp_env_file(
    env_name: str,
    packages_to_install_from_channel: List[str],
    packages_to_install_from_pip: List[str],
) -> tempfile.NamedTemporaryFile:
    """
    create a temporary environment to use at the creation of the mamba env

    Args:
        env_name (str): Name of the env to create
        packages_to_install_from_channel (List[str]): List of packages to install from channel
        packages_to_install_from_pip (List[str]): List of packages to install from pip

    Returns:
        str: Path to the temporary env file
    """

    tmp = tempfile.NamedTemporaryFile(delete=False)

    content = (
        """
name: """
        + env_name
        + """

dependencies:"""
        + "".join([f"\n  - {package}" for package in packages_to_install_from_channel])
    )

    if (
        packages_to_install_from_pip is not None
        and len(packages_to_install_from_pip) > 0
    ):
        content += """
  - pip:""" + "".join(
            [f"\n    - {package}" for package in packages_to_install_from_pip]
        )

    with open(tmp.name, "w") as f:
        f.write(content)

    tmp.close()

    return tmp


def create_custom_env(env_name: str, path_to_env_file: str) -> None:
    """
    create the mamba env for the provided env file

    Args:
        env_name (str): Name of the env to create
        path_to_env_file (str): Path to the env file to use to create the env

    Returns:
        None

    Raises:
        FileNotFoundError: The provided env file couldn't be found
    """
    logger.debug(f"Creating env : {env_name}")

    custom_env = yaml.safe_load(open(path_to_env_file, "r"))

    if "name" in custom_env:
        logger.info(
            f"overwriting default custom env's name {env_name} to {custom_env['name']}"
        )

        env_name = custom_env["name"]

    if "inherit" not in custom_env and "dependencies" not in custom_env:
        error_message = "Provided config env is empty, you must either specify `inherit` or `dependencies`."

        logger.critical(error_message)

        raise RuntimeError(error_message)

    (
        packages_to_install_from_pip,
        packages_to_install_from_channel,
    ) = retrieve_package_from_env_file(custom_env)

    if "inherit" not in custom_env.keys():
        custom_env["inherit"] = []

    for env_to_inherit in custom_env["inherit"]:

        path_to_env_to_inherit_from = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "envs",
            env_to_inherit.split("-")[0],
            "-".join(env_to_inherit.split("-")[1:]) + ".yaml",
        )

        env_file = yaml.safe_load(open(path_to_env_to_inherit_from, "r"))
        pip_packages, channel_packages = retrieve_package_from_env_file(env_file)

        packages_to_install_from_pip += pip_packages

        # install mandatory package gladia-api-utils to handle
        # input and output of the api natively
        packages_to_install_from_pip += f"-e {PATH_TO_GLADIA_SRC}/api_utils/"

        packages_to_install_from_channel += channel_packages

    temporary_file = create_temp_env_file(
        env_name, packages_to_install_from_channel, packages_to_install_from_pip
    )

    os.link(temporary_file.name, temporary_file.name + ".yaml")

    if os.path.isdir(
        os.path.join(MAMBA_ROOT_PREFIX, "envs", env_name)
    ) and not filecmp.cmp(
        temporary_file.name + ".yaml",
        os.path.join(MAMBA_ROOT_PREFIX, "envs", env_name, f"{env_name}.yaml"),
    ):
        try:

            cmd_to_exec = "create"

            if os.path.isdir(os.path.join(MAMBA_ROOT_PREFIX, "envs", env_name)):
                cmd_to_exec = "update"

            subprocess.run(
                f"micromamba {cmd_to_exec} -f {temporary_file.name  + '.yaml'} -y".split(
                    " "
                ),
                check=True,
            )

            subprocess.run(
                f"micromamba clean --all --yes".split(" "),
                check=True,
            )
            logger.info(f"Env {env_name} has been successfully {cmd_to_exec}d")

        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Couldn't create env {env_name}: {error}")

        finally:
            shutil.copyfile(
                src=temporary_file.name + ".yaml",
                dst=os.path.join(
                    MAMBA_ROOT_PREFIX, "envs", env_name, f"{env_name}.yaml"
                ),
            )

            os.remove(temporary_file.name)
            os.remove(temporary_file.name + ".yaml")

    else:
        logger.info(f"Env {env_name} already up to date.")


def build_specific_envs(paths: List[str]) -> None:
    """
    Build mamba envs using the provided {paths}

    Args:
        paths List[str]: List of path to either a model folder or a model's env file (env.yaml)

    Returns:
        None

    Raises:
        FileNotFoundError: The profided model folder or env file couldn't be founded
    """
    paths = set(paths)

    for path in paths:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"custom env {path} not found, please specify a correct path either leading to a model or model's env file."
            )

        if ENV_DEFAULT_FILENAME in path:
            path = os.path.split(path)[0]

        task_path, model = os.path.split(path)
        task = os.path.split(task_path)[1]

        logger.debug(f"building environemnt {task}-{model}")

        create_custom_env(
            env_name=f"{task}-{model}",
            path_to_env_file=os.path.join(path, ENV_DEFAULT_FILENAME),
        )


def build_env_for_activated_tasks(
    path_to_config_file: str, path_to_apis: str, modality=".*", full_path_mode=False
) -> None:
    """
    Build the mamba env for every activated tasks

    Args:
        path_to_config_file (str): Path to the general config file describing which tasks are activated
        path_to_apis (str): Path to the Gladia's tasks
        modality (str): modality name pattern filter (default: .*)
        full_path_mode (bool): If True, will not check regex, not check activated task and
            use modality as a full path to the api env to build (default: False)

    Returns:
        None

    Raises:
        FileNotFoundError: The provided config file couldn't be found
    """

    # if full_path_mode is True, use modality as a full path to the api env to build
    # otherwise, use modality as a regex to filter the activated tasks from the config file
    if full_path_mode:
        logger.debug(f"full_path_mode activated {modality}")
        logger.debug(f"building env for {modality}")

        env_file_path = os.path.join(modality[0], ENV_DEFAULT_FILENAME)

        if os.path.exists(env_file_path):
            head, model = os.path.split(modality[0].rstrip("/"))
            head, task = os.path.split(head.rstrip("/"))

            create_custom_env(
                env_name="-".join([task, model]), path_to_env_file=env_file_path
            )

        else:
            raise FileNotFoundError(
                f"Couldn't find {ENV_DEFAULT_FILENAME} for {modality[0]}, please check your config file."
            )
    else:

        paths = sorted(
            get_activated_task_path(
                path_to_config_file=path_to_config_file, path_to_apis=path_to_apis
            )
        )

        for task in tqdm(paths):

            if not bool(re.search(modality[0], task)):
                logger.debug(f"Skipping task {task}")

                continue

            env_file_path = os.path.join(task, ENV_DEFAULT_FILENAME)
            if os.path.exists(env_file_path):
                create_custom_env(
                    env_name=os.path.split(task)[1],
                    path_to_env_file=env_file_path,
                )

            # make sur we don't have a __pycache__ folder
            # or a file
            models = list(
                filter(
                    lambda dir: os.path.split(dir)[-1][0] not in ["_", "."],
                    os.listdir(task),
                )
            )

            for model in models:
                env_file_path = os.path.join(task, model, ENV_DEFAULT_FILENAME)
                if not os.path.exists(env_file_path):
                    continue

                create_custom_env(
                    env_name=f"{os.path.split(task)[-1]}-{model}",
                    path_to_env_file=env_file_path,
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        action="append",
        type=str,
        help="Specify the name of a specific env to build. You can define this arg multiple time to build multiple specific envs.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        type=str,
        help="Specify a RegExp to filter input nd output modalities to process. If full_path_mode set to True a full_path is expected. default .*",
    )
    parser.add_argument(
        "--path_to_apis",
        action="append",
        type=str,
        help="Specify a path to the api app .*",
    )
    parser.add_argument(
        "--debug_mode",
        dest="debug_mode",
        action="store_true",
        default=False,
        help="Activate the debug mode for logger (True if called)",
    )
    parser.add_argument(
        "--full_path_mode",
        dest="full_path_mode",
        action="store_true",
        default=False,
        help="Activate the strict mode for modality/task/model path (True if called)",
    )
    args = parser.parse_args()

    if args.name:
        return build_specific_envs(args.name)

    if args.path_to_apis:
        path_to_apis = args.path_to_apis
    else:
        path_to_apis = os.path.join(os.getenv("PATH_TO_GLADIA_SRC", "/app"), "apis")

    if args.debug_mode:
        logger.setLevel(logging.DEBUG)

    return build_env_for_activated_tasks(
        path_to_config_file=os.path.join(path_to_apis, "..", "config.json"),
        path_to_apis=path_to_apis,
        modality=args.modality,
        full_path_mode=args.full_path_mode,
    )


if __name__ == "__main__":
    main()

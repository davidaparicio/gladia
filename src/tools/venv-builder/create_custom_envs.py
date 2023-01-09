import argparse
import difflib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import yaml
from tqdm import tqdm

logger = getLogger(__name__)

LOGGER_COLOR_CYAN = "\x1b[36m"
LOGGER_COLOR_RESET = "\x1b[39m"

ENV_DEFAULT_FILENAME_WITHOUT_EXTENSION = "env"
ENV_DEFAULT_FILENAME = f"{ENV_DEFAULT_FILENAME_WITHOUT_EXTENSION}.yaml"

PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app/src")
GLADIA_PERSISTENT_PATH = os.getenv("GLADIA_PERSISTENT_PATH", "/gladia")
MAMBA_ROOT_PREFIX = os.getenv("MAMBA_ROOT_PREFIX", f"{GLADIA_PERSISTENT_PATH}/conda")

FORCE_ENV_UPDATE = os.getenv("FORCE_ENV_UPDATE", "False").lower() == "true"


def delete_file(file_path: str) -> None:
    """
    Delete a file if it exists

    Args:
        file_path (str): path to the file to delete

    Returns:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_gladia_api_utils_package() -> Tuple[List[str], List[str]]:
    """
    Retrieve the packages to install from the gladia api utils env file
    and return them in two lists, one for pip packages and one for mamba packages

    Args:
        None

    Returns:
        Tuple[List[str], List[str]]: a tuple containing two lists, one for pip packages and one for mamba packages
    """
    gladia_api_utils_env_path = f"{PATH_TO_GLADIA_SRC}/api_utils/env.txt"
    pip_packages_list = []
    mamba_packages_list = []
    # open the env file and retrieve the packages to install
    # each line represent a package to install
    # if the package starts with a pip::, it means it should be installed from pip
    # it means it should be in the pip_packages_list
    # otherwise it should be in the mamba_packages_list
    with open(gladia_api_utils_env_path, "r") as f:
        for line in f.readlines():
            if line.startswith("pip::"):
                pip_packages_list.append(line.replace("pip::", "").strip())
            else:
                mamba_packages_list.append(line.strip())

    return pip_packages_list, mamba_packages_list


def diff_lines(file1: str, file2: str) -> List[str]:
    """
    Compare two files line by line and return the lines that are different.

    Args:
        file1 (str): path to the first file
        file2 (str): path to the second file

    Returns:
        List[str]: list of the lines that are different between the two files
    """
    with open(file1, "r") as f1:
        lines1 = f1.readlines()
    with open(file2, "r") as f2:
        lines2 = f2.readlines()

    diff = difflib.ndiff(lines1, lines2)

    return [
        line
        for line in diff
        if (
            (line.startswith("-") or line.startswith("+"))
            and line != "+ \n"
            and line != "- \n"
        )
    ]


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


def create_temp_env_files(
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
        str: Path to the temporary env package only file
        str: Path to the temporary env pip only file
    """

    tmp, tmpchan, tmppip = (
        tempfile.NamedTemporaryFile(delete=False),
        tempfile.NamedTemporaryFile(delete=False),
        tempfile.NamedTemporaryFile(delete=False),
    )

    content_chan = (
        """
name: """
        + env_name
        + """

dependencies:"""
        + "".join([f"\n  - {package}" for package in packages_to_install_from_channel])
    )

    content = content_chan
    content_pip = str()
    content_pip_requirements = str()

    if (
        packages_to_install_from_pip is not None
        and len(packages_to_install_from_pip) > 0
    ):
        content_pip = "".join(
            [f"\n    - {package}" for package in packages_to_install_from_pip]
        )

        content_pip_requirements = "".join(
            [f"\n{package}" for package in packages_to_install_from_pip]
        )

        content += (
            """
  - pip:"""
            + content_pip
        )

    with open(tmp.name, "w") as f:
        f.write(content)

    tmp.close()

    with open(tmpchan.name, "w") as f:
        f.write(content_chan)

    tmpchan.close()

    with open(tmppip.name, "w") as f:
        f.write(content_pip_requirements)

    tmppip.close()

    return tmp, tmpchan, tmppip


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
            f"Overwriting default custom env's name {env_name} to {custom_env['name']}"
        )

        env_name = custom_env["name"]

    if "inherit" not in custom_env and "dependencies" not in custom_env:
        error_message = "Provided config env is empty, you must either specify `inherit` or `dependencies`."

        logger.critical(error_message)

        raise RuntimeError(error_message)

    # in order we will get the packages from the
    # gladia_api_utils then install
    # the inherated packages from the env
    # and finally install the packages from the env file
    list_of_pip_packages_to_install = list()
    list_of_channel_packages_to_install = list()

    # get the gladia_api_utils_packages
    # this is needed to be able to use the gladia_api_utils
    # especially to handle the input and output of the tasks
    # as well as other utils like gpu / cuda handling
    (
        gladia_api_utils_pip_packages,
        gladia_api_utils_channel_packages,
    ) = get_gladia_api_utils_package()

    list_of_pip_packages_to_install += gladia_api_utils_pip_packages
    list_of_channel_packages_to_install += gladia_api_utils_channel_packages

    # get the packages from the env to inherit from
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
        (
            inherated_pip_packages,
            inherated_channel_packages,
        ) = retrieve_package_from_env_file(env_file)

        list_of_pip_packages_to_install += inherated_pip_packages
        list_of_channel_packages_to_install += inherated_channel_packages

    (
        packages_to_install_from_pip,
        packages_to_install_from_channel,
    ) = retrieve_package_from_env_file(custom_env)

    list_of_pip_packages_to_install += packages_to_install_from_pip
    list_of_channel_packages_to_install += packages_to_install_from_channel

    # install mandatory package gladia-api-utils to handle
    # input and output of the api natively
    packages_to_install_from_pip += [f"-e {PATH_TO_GLADIA_SRC}/api_utils/"]

    temporary_file, temporary_file_channel, temporary_file_pip = create_temp_env_files(
        env_name, packages_to_install_from_channel, list_of_pip_packages_to_install
    )

    final_env_file_path = os.path.join(
        MAMBA_ROOT_PREFIX, "envs", env_name, ENV_DEFAULT_FILENAME
    )

    final_env_channel_file_path = os.path.join(
        MAMBA_ROOT_PREFIX, "envs", env_name, f"{ENV_DEFAULT_FILENAME_WITHOUT_EXTENSION}-channel.yaml"
    )

    final_env_pip_file_path = os.path.join(
        MAMBA_ROOT_PREFIX, "envs", env_name, f"{ENV_DEFAULT_FILENAME_WITHOUT_EXTENSION}-pip.txt"
    )

    try:

        # check if final_env_file_path exists
        final_env_file_path_exists = os.path.isfile(final_env_file_path)
        final_env_channel_file_path_exists = os.path.isfile(final_env_channel_file_path)
        final_env_pip_file_path_exists = os.path.isfile(final_env_pip_file_path)

        action = ""

        # create a temp yaml file to be able to build the env with micromamba
        temp_env_file_path = temporary_file.name + ".yaml"
        os.link(temporary_file.name, temp_env_file_path)

        # check if env need to be updated
        # if FORCE_ENV_UPDATE is set to true
        # or if the env file has changed

        # first check if the env already exists
        should_update_pip = False
        should_update_channel = False
        if final_env_file_path_exists:
            # if FORCE_ENV_UPDATE is set to true we will update the env
            if FORCE_ENV_UPDATE:
                logger.info(
                    LOGGER_COLOR_CYAN
                    + f"{LOGGER_COLOR_CYAN}Env {env_name}: env force to update by FORCE_ENV_UPDATE env. variable or script flag --force_update\n to avoid force update set $ export FORCE_ENV_UPDATE=false"
                    + LOGGER_COLOR_RESET
                )
                action = "update"
                should_update_channel = True
                should_update_pip = True

            # if the env file has changed we will update the env
            elif len(diff_lines(final_env_file_path, temp_env_file_path)) > 0:

                if (
                    final_env_channel_file_path_exists
                    and len(
                        diff_lines(
                            final_env_channel_file_path, temporary_file_channel.name
                        )
                    )
                    > 0
                ):
                    logger.info(
                        LOGGER_COLOR_CYAN
                        + f"Env {env_name}: env's channel dependencies changed meaning env's channel need a rebuild"
                        + LOGGER_COLOR_RESET
                    )
                    action = "update"
                    should_update_channel = True

                if (
                    final_env_pip_file_path_exists
                    and len(
                        diff_lines(final_env_pip_file_path, temporary_file_pip.name)
                    )
                    > 0
                ):
                    logger.info(
                        LOGGER_COLOR_CYAN
                        + f"Env {env_name}: env's pip dependencies changed meaning env's pip need a rebuild"
                        + LOGGER_COLOR_RESET
                    )
                    action = "update"
                    should_update_pip = True

        # if the env doesn't exist we will create it
        else:
            logger.info(
                LOGGER_COLOR_CYAN
                + f"Env {env_name}: env file doesn't exist meaning the env was never fully initiated"
                + LOGGER_COLOR_RESET
            )
            action = "create"

        # lets apply the env action (update or create)
        # action = "" means no action needed
        if action != "":
            logger.info(
                LOGGER_COLOR_CYAN
                + f"Env {env_name} will be {action}d"
                + LOGGER_COLOR_RESET
            )

            if action == "create":
                logger.info(
                    LOGGER_COLOR_CYAN + f"Creating env {env_name}" + LOGGER_COLOR_RESET
                )

                subprocess.run(
                    [
                        "micromamba",
                        "create",
                        "-f",
                        f"{temporary_file.name}.yaml",
                        "--retry-clean-cache",
                        "-y",
                    ]
                )

            # update the env's pip packages if needed
            # we will check if the action is update
            # if so we should have a pre-existing pip env file
            # if this pre-existing pip env file is not empty and is different from the new one
            # we will update the to env completely
            # else if not different from the previous one and that FORCE_ENV_UPDATE is set to true
            # if the env is being created we don't need to check if the pip env file is different
            if action == "update":
                logger.info(
                    LOGGER_COLOR_CYAN + f"Updating env {env_name}" + LOGGER_COLOR_RESET
                )
                if should_update_channel:
                    logger.info(
                        LOGGER_COLOR_CYAN
                        + f"Updating channels for env {env_name}"
                        + LOGGER_COLOR_RESET
                    )
                    os.link(
                        temporary_file_channel.name,
                        temporary_file_channel.name + ".yaml",
                    )

                    subprocess.run(
                        [
                            "micromamba",
                            "install",
                            "-f",
                            f"{temporary_file_channel.name}.yaml",
                            "--retry-clean-cache",
                            "-y",
                        ]
                    )

                if should_update_pip:
                    logger.info(
                        LOGGER_COLOR_CYAN
                        + f"Updating pip for env {env_name}"
                        + LOGGER_COLOR_RESET
                    )
                    os.link(temporary_file_pip.name, temporary_file_pip.name + ".txt")
                    subprocess.run(
                        [
                            "micromamba",
                            "run",
                            "-n",
                            env_name,
                            "/bin/bash",
                            "-c",
                            f"pip install --upgrade -r {temporary_file_pip.name}.txt",
                        ]
                    )

        # if action = "" just log that the env is up to date
        else:
            logger.info(
                LOGGER_COLOR_CYAN
                + f"Env {env_name}: env file didn't change and FORCE_ENV_UPDATE=false meaning env doesn't need a rebuild however if your face any issue try to set $ export FORCE_ENV_UPDATE=true to force the upgrade of packages as they were skipped for performance purpose"
                + LOGGER_COLOR_RESET
            )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Couldn't create env {env_name}: {error}")

    finally:
        shutil.copyfile(src=temporary_file.name, dst=final_env_file_path)

        shutil.copyfile(
            src=temporary_file_channel.name,
            dst=final_env_channel_file_path,
        )

        if os.stat(temporary_file_pip.name).st_size != 0:
            shutil.copyfile(
                src=temporary_file_pip.name,
                dst=final_env_pip_file_path,
            )
            Path(temporary_file_pip.name)

        delete_file(temp_env_file_path)
        delete_file(temporary_file.name)
        delete_file(temporary_file.name + ".yaml")
        delete_file(temporary_file_channel.name)
        delete_file(temporary_file_channel.name + ".yaml")
        delete_file(temporary_file_pip.name)
        delete_file(temporary_file_pip.name + ".txt")


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
                f"Custom env {path} not found, please specify a correct path either leading to a model or model's env file."
            )

        if ENV_DEFAULT_FILENAME in path:
            path = os.path.split(path)[0]

        task_path, model = os.path.split(path)
        task = os.path.split(task_path)[1]

        logger.debug(f"Building environemnt {task}-{model}")

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
    parser.add_argument(
        "--server_env",
        dest="server_env",
        action="store_true",
        default=False,
        help="Build the server env only (True if called)",
    )
    parser.add_argument(
        "--force_update",
        dest="force_update",
        action="store_true",
        default=False,
        help="Force update of the env (True if called)",
    )
    args = parser.parse_args()

    if args.debug_mode:
        # Set the logger level to DEBUG
        logger.setLevel(logging.DEBUG)

        # Create a logger handler that writes log messages to the console
        console_handler = logging.StreamHandler()

        # Set the logger handler level to DEBUG
        console_handler.setLevel(logging.DEBUG)

        # Add the logger handler to the logger
        logger.addHandler(console_handler)

    if args.force_update:
        global FORCE_ENV_UPDATE
        FORCE_ENV_UPDATE = True

    if args.name:
        return build_specific_envs(args.name)

    if args.path_to_apis:
        path_to_apis = args.path_to_apis
    else:
        path_to_apis = os.path.join(os.getenv("PATH_TO_GLADIA_SRC", "/app"), "apis")

    if args.server_env:
        path_to_server_env = os.path.join(
            os.getenv("PATH_TO_GLADIA_SRC", "/app"), "env.yaml"
        )

        env = create_custom_env(
            env_name="server",
            path_to_env_file=path_to_server_env,
        )

        # install extra packages for the server
        subprocess.run(
            [
                "micromamba",
                "run",
                "-n",
                "server",
                "/bin/bash",
                "-c",
                'pip install "jax[cuda11_cudnn82]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
            ]
        )

        return env
    else:
        # avoid importing this api_utils at the top of the file
        # to avoid issue at the init of the server env
        global get_activated_task_path
        from gladia_api_utils import get_activated_task_path

        return build_env_for_activated_tasks(
            path_to_config_file=os.path.join(path_to_apis, "..", "config.json"),
            path_to_apis=path_to_apis,
            modality=args.modality,
            full_path_mode=args.full_path_mode,
        )


if __name__ == "__main__":
    main()

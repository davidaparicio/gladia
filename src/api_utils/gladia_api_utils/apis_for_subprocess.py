import json
import os
import socket
import subprocess
import time
from asyncio.log import logger
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from urllib import request

import requests

logger = getLogger(__name__)

SERVICE_PATH = "/etc/supervisor/conf.d/"
LOGS_PATH = "/var/log/supervisor/"


def __create_subprocess_api_config(api_name_to_build: str) -> Tuple[str, str, str, int]:
    """
    Build the command to start a api for a subprocess

    Args:
        api_name_to_build (str): The name of the api to build
        (should look like /input/output/task/model-name/)
        (example "/text/text/language-detection/toftrup-etal-2021/")
        (should be the same as the api name in the config.json file)

    Returns:
        Tuple[str, str, str, int]: Tuple of the api config : subprocess_api_dir, micromamba_env_name, model_name, port
    """
    # 1. get the GLADIA_SRC environment variable
    path_to_gladia_src = os.environ.get("PATH_TO_GLADIA_SRC", "/app")

    # 2. Get the path to the subprocess api
    subprocess_api_dir = os.path.join(path_to_gladia_src, "apis") + api_name_to_build

    # 3. Create a command string to build the subprocess api
    # using micromamba and the subprocess api's micromamba env
    # built previously and name as follows:
    # task-models-model-name
    # example: image-generation-models-dream-studio
    # where the api name is /input/output/image-generation/dream-studio/
    # (see create_custom_envs.py)

    # 3.1. get the name of the micromamba env
    # (example: image-generation-models-dream-studio)
    micromamba_env_name = "-".join(api_name_to_build.split("/")[3:-1])
    model_name = api_name_to_build.split("/")[4]
    port = __get_api_port(api_name_to_build)

    return subprocess_api_dir, micromamba_env_name, model_name, port


def __create_unit_file_for_api(api_name: str) -> str:
    """
    Create a supervisord unit file for a subprocess api

    Args:
        api_name (str): Name of the api to create the unit file for

    Returns:
        str: Service name
    """

    # 1. get the config used to start the api
    (
        subprocess_api_dir,
        micromamba_env_name,
        model_name,
        port,
    ) = __create_subprocess_api_config(api_name)

    service_name = micromamba_env_name

    # 2. get the path to the supervisor service
    service_file_path = os.path.join(SERVICE_PATH, f"{service_name}.conf")
    cwd = os.path.dirname(os.path.abspath(__file__))
    mamba_root_prefix = os.getenv("MAMBA_ROOT_PREFIX", "/opt/conda")

    # 3. create the log file
    log_path = os.path.join(LOGS_PATH, f"{service_name}")
    Path(log_path).touch()

    # 3. create the unit file
    unit_file = f"""
[supervisord]
nodaemon=false

[program:{service_name}]
user=root
directory={cwd}
command=/usr/local/bin/micromamba run -r {mamba_root_prefix} -n {micromamba_env_name} --cwd={subprocess_api_dir} python3 {cwd}/fastapi_runner.py {port} {subprocess_api_dir}/{model_name}.py
stdout_logfile={log_path}
stdout_logfile_maxbytes=0
stderr_logfile={log_path}
stderr_logfile_maxbytes=0
startsecs=5
stopwaitsecs=5
autostart=false
autorestart=true
"""

    # 4. write the unit file
    with open(service_file_path, "w") as f:
        f.write(unit_file)

    return service_name


def __reload_supervisord_daemon() -> None:
    """
    Reload the supervisord daemon

    Args:
        None

    Returns:
        None
    """
    subprocess.run(["supervisorctl", "reread"])


def __enable_api_service() -> None:
    """
    Enable a supervisord service for a subprocess api

    Args:
        None

    Returns:
        None
    """

    subprocess.run(["supervisorctl", "update"])


def __start_api_service(service_name: str) -> None:
    """
    Start a supervisord service for a subprocess api

    Args:
        service_name (str): Name of the api to start the service for

    Returns:
        None
    """

    subprocess.run(["supervisorctl", "start", service_name])


def __clean_name_for_env_var(name: str) -> str:
    """
    Clean a name to be used as an environment variable

    Args:
        name (str): Name to clean

    Returns:
        str: Cleaned name
    """
    return name.replace("-", "").replace("/", "").replace("_", "").upper()


def __get_api_port(api_name: str) -> Union[int, bool]:
    """
    Get the port of a subprocess api based on it's name

    Args:
        api_name (str): Name of the subprocess api to get the port of
        (should look like "/text/text/language-detection/toftrup-etal-2021/")

    Returns:
        Union[int, bool]: Port of the subprocess api if it exists, False otherwise
    """

    port = os.getenv(__clean_name_for_env_var(api_name), False)

    if not port:
        return False
    else:
        return int(port)


def __find_free_port() -> int:
    """
    Starts a socket connection to grab a free port (Involves a race
        condition but will do for now)

    Args:
        None

    Returns:
        int: Free port
    """
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()

    return int(port)


def __set_api_port(api_name: str) -> int:
    """
    Set the port of an api in the environment variables

    Args:
        api_name (str): Name of the api to set the port for
        (should look like /input/output/task/model-name/)

    Returns:
        int: Port of the api
    """
    port = __find_free_port()

    logger.info(f"Setting port for {__clean_name_for_env_var(api_name)} to {port}")

    os.environ[__clean_name_for_env_var(api_name)] = str(port)

    return port


def __get_subprocess_apis_to_start_from_config() -> Dict[str, int]:
    """
    Get the list of subprocess apis to start from the config file

    Args:
        None

    Returns:
        Dict[str, int]: Dict of subprocess apis to start along with their port
    """
    # 1. get the config.json file path from the environment variable
    path_to_gladia_src = os.environ.get("PATH_TO_GLADIA_SRC", "/app")
    api_config_file = os.getenv("API_CONFIG_FILE", "config.json")
    config_file_path = os.path.join(path_to_gladia_src, api_config_file)

    # 2. load the config file
    with open(config_file_path, "r") as f:
        config = json.load(f)

    # 3. get the list of apis to start from the config.json file using the subprocess_to_apify key
    apis_to_start = config.get("subprocess_to_apify", [])

    # 4. set os environment variable to the list of apis to start
    apis_to_start_dict = dict()
    for api in apis_to_start:
        # calculate the hash api name and truncate to 5 digits to
        # set a port name for the api
        port = __set_api_port(api)
        apis_to_start_dict[api] = port

    # 4. return the list of apis to start as a list of strings
    return apis_to_start_dict


def call_subprocess_api(api_name: str, kwargs: dict()) -> Any:
    """
    Call a subprocess api and return the response

    Args:
        api_name (str): Name of the subprocess api to call
        (should look like "/text/text/language-detection/toftrup-etal-2021/")
        kwargs (dict): Dict of kwargs to pass to the subprocess api

    Returns:
        Any: Response from the subprocess api
    """
    # 1. get the host and port of the subprocess api based on the api name in the environment variables
    port = __get_api_port(api_name)

    # 2. if the port is not set, return False
    if port:
        api_url = f"http://localhost:{port}/predict"

        # 3. call the subprocess api and return the response
        logger.info(f"Calling subprocess api {api_name} at {api_url}")

        headers = {
            "accept": "application/json",
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }

        params = {}

        response = requests.post(api_url, params=params, headers=headers, files=kwargs)

        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                try:
                    return response.content
                except Exception:
                    logger.error(
                        f"Could not parse response from subprocess api {api_name} at {api_url} falling back to raw response"
                    )
                    return response.text
        else:
            logger.error(
                f"Error calling subprocess api {api_name} at {api_url}: {response.status_code}"
            )
            return False

    else:
        return False


def is_subprocess_api_running(api_name_to_check: str) -> bool:
    """

    Check if a subprocess api is running based on it's name

    Args:
        api_name_to_check (str): The name of the api to check
        (should look like "/text/text/language-detection/toftrup-etal-2021/")

    Returns:
        bool: True if the api is running, False otherwise
    """
    # 1. get the host and port of the subprocess api based on the api name in the environment variables
    port = __get_api_port(__clean_name_for_env_var(api_name_to_check))

    # 2. if the port is not set, return False
    if port:
        api_url = f"http://localhost:{port}/status"

        # 3. try to get the status of the api
        try:
            response = request.urlopen(api_url)
            response = json.loads(response.read())
            return True
        except Exception as e:
            logger.error(
                f"Could not get the status of the subprocess api {api_name_to_check} at {api_url}: {e}"
            )
            return False
    else:
        return False


def build_all_subprocesses_apis() -> Dict[int, bool]:
    """
    Build all subprocess apis

    Args:
        None

    Returns:
        Dict[int, bool]: Dict of subprocess apis to start along with their port
    """
    apis_to_start = __get_subprocess_apis_to_start_from_config()

    # create a the folder to store the logs of the subprocess apis
    # if not exists
    os.makedirs(LOGS_PATH, exist_ok=True)

    apis_started = {}
    for api_name, api_port in apis_to_start.items():
        service_name = __create_unit_file_for_api(api_name)
        __reload_supervisord_daemon()
        __enable_api_service()
        __start_api_service(service_name)
        time.sleep(2)
        apis_started[api_name] = {
            "port": api_port,
            "running": is_subprocess_api_running(api_name),
            "service_name": service_name,
        }

        logger.info(f"apis_started: {apis_started}")

    return apis_started

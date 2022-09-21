import importlib
import importlib.util
import json
import logging
import os
import pkgutil
import sys
from distutils.command.clean import clean
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
from os.path import basename, normpath
from types import ModuleType
from typing import List

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi_utils.timing import add_timing_middleware
from gladia_api_utils.apis_for_subprocess import build_all_subprocesses_apis
from gladia_api_utils.submodules import to_task_name
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import RedirectResponse

apis_folder_name = "apis"

import apis


def __init_config() -> dict:
    """
    Load config file and return it as a dict.
    Default path is `config.json`, use API_CONFIG_FILE varenv to change it.

    Args:
        None

    Returns:
        dict: config dict
    """

    config_file = os.getenv("API_CONFIG_FILE", "config.json")

    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            return json.load(f)


def __init_logging(api_config: dict) -> logging.Logger:
    """
    Create a logging.Logger with it format set to config["logs"]["log_format"] f exist, else default.

    Args:
        api_config (dict): config dict

    Returns:
        logging.Logger: logger
    """

    logging_level = {
        None: logging.NOTSET,
        "": logging.NOTSET,
        "none": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(api_config["logs"]["log_level"], logging.INFO)

    logging.basicConfig(
        level=logging_level,
        format=api_config["logs"]["log_format"],
    )

    logger = logging.getLogger(__name__)

    rotating_file_handler = RotatingFileHandler(
        api_config["logs"]["log_path"],
        maxBytes=100_000,
        backupCount=10,
    )
    rotating_file_handler.setFormatter(
        logging.Formatter(api_config["logs"]["log_format"])
    )
    rotating_file_handler.setLevel(logging_level)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(logging.Formatter(api_config["logs"]["log_format"]))
    stream_handler.setLevel(logging_level)

    logger.addHandler(rotating_file_handler)
    logger.addHandler(stream_handler)

    existing_loggers = [logging.getLogger()]
    existing_loggers += [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    set_logger_handlers = set(logger.handlers)

    for external_logger in existing_loggers:
        if "gladia_api_utils" in external_logger.name:
            continue

        for handler in external_logger.handlers:
            handler.setLevel(logging_level)

        # Prevents having multiple times the same handler
        external_logger.handlers = list(
            set(external_logger.handlers).union(set(set_logger_handlers))
        )

    return logger


def __init_prometheus_instrumentator(instrumentator_config: dict) -> Instrumentator:
    """
    Initialize the prometheus_fastapi_instrumentator.Instrumentator using api config dict

    Args:
        instrumentator_config (dict): config dict

    Returns:
        Instrumentator: Initialized Instrumentator
    """

    return Instrumentator(
        should_group_status_codes=instrumentator_config["should_group_status_codes"],
        should_ignore_untemplated=instrumentator_config["should_ignore_untemplated"],
        should_group_untemplated=instrumentator_config["should_group_untemplated"],
        should_respect_env_var=instrumentator_config["should_respect_env_var"],
        env_var_name=instrumentator_config["env_var_name"],
        excluded_handlers=instrumentator_config["excluded_handlers"],
        should_round_latency_decimals=instrumentator_config[
            "should_round_latency_decimals"
        ],
        round_latency_decimals=instrumentator_config["round_latency_decimals"],
        should_instrument_requests_inprogress=instrumentator_config[
            "should_instrument_requests_inprogress"
        ],
        inprogress_name=instrumentator_config["inprogress_name"],
        inprogress_labels=instrumentator_config["inprogress_labels"],
    )


def __set_app_middlewares(api_app: FastAPI, api_config: dict) -> None:
    """
    Set up the api middlewares

    Args:
        api_app (FastAPI): FastAPI representing the API
        api_config (dict): config dict telling which middlewares to use

    Returns:
        None
    """

    if api_config["logs"]["timing_activated"]:
        add_timing_middleware(api_app, record=logging.info, prefix="app")

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config["CORS"]["allow_origins"],
        allow_credentials=api_config["CORS"]["allow_credentials"],
        allow_methods=api_config["CORS"]["allow_methods"],
        allow_headers=api_config["CORS"]["allow_headers"],
    )


def __add_router(module: ModuleType, module_path: str) -> None:
    """
    Add the module router to the API app

    Args:
        module (ModuleType): module to add to the API app
        module_path (str): module path

    Returns:
        None
    """

    # remove the "apis" part of the path
    module_input, module_output, module_task = module_path.replace(
        apis_folder_name, ""
    )[1:].split(".")

    module_task = to_task_name(module_task).upper()
    module_config = config["active_tasks"][module_input][module_output]

    active_task_list = list(map(lambda each: to_task_name(each).upper(), module_config))

    if "NONE" not in active_task_list and (
        module_task in active_task_list or "*" in module_config
    ):
        # remove the "apis" part of the path
        module_prefix = module_path.replace(".", "/").replace(apis_folder_name, "")
        app.include_router(module.router, prefix=module_prefix)


def __clean_package_import(module_path: str) -> ModuleType:
    """
    import package based on path and create an alias for it if needed
    to avoid import errors when path contains hyphens
    Args:
        module_path (str): path to the package to import
    Returns:
        ModuleType: imported package
    """
    clean_key = module_path.replace("-", "_")
    module = importlib.import_module(module_path)
    # clean_key is used to avoid importlib.import_module to import the same module twice
    # if the module is imported twice, the second import will fail
    # this is a workaround to avoid this issue
    # see https://stackoverflow.com/questions/8350853/how-to-import-module-when-module-name-has-a-dash-or-hyphen-in-it
    if clean_key not in sys.modules:
        sys.modules[clean_key] = sys.modules[module_path]
    return module


def __module_is_an_input_type(split_module_path: list) -> bool:
    """
    Check if the parsed module_path is an input type (Image/Audio/Video/Text)
    (meaning length is 1)

    Args:
        split_module_path (list): module path split by "."

    Returns:
        bool: True if the module is an input type, False otherwise
    """
    return len(split_module_path) == 1


def __module_is_a_modality(split_module_path: list, module_config: dict) -> bool:
    """
    Check if the module is a modality could be an input or output type
    with values like image, text, etc.

    Args:
        split_module_path (list): module path split by "."
        module_config (dict): module config dict

    Returns:
        bool: True if the module is a modality, False otherwise
    """
    return (
        len(split_module_path) == 2
        and "None".upper not in map(lambda each: each.upper(), module_config)
        or len(module_config) == 0
    )


def __module_is_a_task(split_module_path: List[str], module_config: dict) -> bool:
    """
    Check if the module is a task with values like classification, detection, etc.

    Args:
        split_module_path (list): module path split by "."
        module_config (dict): module config dict

    Returns:
        bool: True if the module is a task, False otherwise
    """
    return (
        len(split_module_path) == 2
        and "None".upper not in map(lambda each: each.upper(), module_config)
        or len(module_config) == 0
    )


def __module_is_a_model(split_module_path: List[str]) -> bool:
    """
    Check if the module is a model with values like inception, resnet, etc.

    Args:
        split_module_path (list): module path split by "."

    Returns:
        bool: True if the module is a model, False otherwise
    """
    return len(split_module_path) == 4


def __module_is_subprocess(module_path: str) -> bool:
    """
    Check if the module is a subprocess looking for env.yaml file within
    the module path

    Args:
        module_path (str): module path

    Returns:
        bool: True if the module is a subprocess, False otherwise
    """
    # check if a env.yaml file exist in the module path
    # if so it is a subprocess : return True
    return os.path.exists(os.path.join(module_path, "env.yaml"))


def import_submodules(package: ModuleType, recursive: bool = True) -> None:
    """
    Import every task presents in the API by loading each submodule (recursively by default)

    Args:
        package (module): root package to import every submodule from (usually: apis)
        recursive (bool): if True, import every submodule recursively (default True)

    Returns:
        None
    """

    if isinstance(package, str):
        package = __clean_package_import(package)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):

        module_path = f"{package.__name__}.{name}"

        # get back the module file path from name
        # replacing the . with /
        # also make the path absolute
        module_file_path = os.path.abspath(module_path.replace(".", "/"))

        if not __module_is_subprocess(module_file_path):
            module = __clean_package_import(module_path)

        module_relative_path = module_path.replace("apis", "")[1:]

        if "module" in vars() and "router" in dir(module):
            __add_router(module, module_path)

        if not recursive or not is_pkg:
            continue

        module_split = module_relative_path.split(".")
        module_config = (
            config["active_tasks"][module_split[0]][module_split[1]]
            if len(module_split) > 1
            else []
        )

        if (
            __module_is_an_input_type(module_split)
            or __module_is_a_modality(module_split, module_config)
            or __module_is_a_task(module_split, module_config)
            or __module_is_a_model(module_split)
        ) and (not __module_is_subprocess(module_file_path)):
            import_submodules(module_path)
        else:
            logger.debug(f"skipping {module_relative_path}")


nltk.download("punkt")

os.environ["TRITON_MODELS_PATH"] = os.getenv(
    "TRITON_MODELS_PATH", default="/tmp/gladia/triton"
)

config = __init_config()
logger = __init_logging(config)

app = FastAPI(default_response_class=ORJSONResponse)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


__set_app_middlewares(app, config)

if config["prometheus"]["active"]:
    instrumentator = __init_prometheus_instrumentator(
        config["prometheus"]["instrumentator"]
    )


build_all_subprocesses_apis()

import_submodules(apis)

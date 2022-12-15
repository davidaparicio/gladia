import importlib
import logging
import os
import pkgutil
import sys
from logging import getLogger
from types import ModuleType
from typing import Any, Dict, List
from warnings import warn

from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

logger = getLogger(name=__name__)


def __add_router(
    app,
    module: ModuleType,
    module_path: str,
    active_tasks: Dict[str, Any],
    apis_folder_name: str,
) -> None:
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

    module_config = active_tasks[module_input][module_output]

    active_task_list = set(map(lambda each: each.split("?")[0], module_config))

    if (
        ("NONE" not in active_task_list and "NONE" not in module_config)
        and (module_task in active_task_list or "*" in module_config)
        and (module_path[:4] == "apis")
    ):
        task_metadata = get_task_metadata(module_path.replace(".", "/"))

        inputs = [
            {
                "name": input_name,
                "type": task_metadata["inputs"][input_name]["type"],
                "default": task_metadata["inputs"][input_name].get("default", ...),
                "example": task_metadata["inputs"][input_name]["examples"][0],
                "examples": task_metadata["inputs"][input_name]["examples"],
                "placeholder": task_metadata["inputs"][input_name]["placeholder"],
            }
            for input_name in task_metadata["inputs"]
        ]

        output = {
            "name": task_metadata["output"]["name"],
            "type": task_metadata["output"]["type"],
            "example": task_metadata["output"]["example"],
        }

        router = APIRouter()

        default_model = task_metadata["default-model"]

        if "default-model-version" in task_metadata:
            default_model_version = task_metadata["default-model-version"]
        else:
            default_model_version = None

        task_router = TaskRouter(
            router=router,
            input=inputs,
            output=output,
            default_model=default_model,
            default_model_version=default_model_version,
            rel_path=module_path.replace(".", "/"),
        )

        if os.getenv("LAZY_WARM_UP", "true").lower() == "false":
            task_router.prepare()
        else:
            warn(
                "LAZY_WARM_UP is set to true, this could result in a long first call to the model in order to init it."
            )

        module_prefix = module_path.replace(".", "/").replace(apis_folder_name, "")

        app.include_router(router, prefix=module_prefix)  #


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
        len(split_module_path) == 3
        and "None".upper not in map(lambda each: each.upper(), module_config)
        or len(module_config) == 0
    )


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


def add_routes_to_router(
    app,
    apis_folder_name: str,
    active_tasks: Dict[str, Any],
    package: ModuleType,
    recursive: bool = True,
) -> None:
    """
    Import every task presents in the API by loading each submodule (recursively by default)

    Args:
        app: app server to add the routes to
        apis_folder_name (str): folder where all the apis are stored  (usually: apis/)
        active_tasks (Dict[str, Any]): list of the active tasks
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

        if (
            "module" in vars()
            and len(module_path.split(".")) == 4
            and "task.yaml"
            in os.listdir("apis/" + module_relative_path.replace(".", "/"))
        ):
            __add_router(app, module, module_path, active_tasks, apis_folder_name)

        if not recursive or not is_pkg:
            continue

        module_split = module_relative_path.split(".")
        module_config = (
            active_tasks[module_split[0]][module_split[1]]
            if len(module_split) > 1
            else []
        )

        if (
            __module_is_an_input_type(module_split)
            or __module_is_a_modality(module_split, module_config)
            or __module_is_a_task(module_split, module_config)
        ) and (not __module_is_subprocess(module_file_path)):
            add_routes_to_router(app, apis_folder_name, active_tasks, module_path)
        else:
            logging.debug(f"skipping {module_relative_path}")

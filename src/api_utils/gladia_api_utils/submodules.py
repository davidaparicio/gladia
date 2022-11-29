import importlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from enum import Enum, EnumMeta
from logging import getLogger
from pathlib import Path
from shlex import quote
from time import time
from typing import Any, List, Optional, Tuple, Union
from urllib.request import urlopen

import forge
import starlette
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from gladia_api_utils.apis_for_subprocess import (
    call_subprocess_api,
    is_subprocess_api_running,
)
from pydantic import BaseModel, create_model

from .casting import cast_response
from .file_management import is_binary_file, is_valid_path, write_tmp_file
from .responses import AudioResponse, ImageResponse, VideoResponse

versions = list()
available_versions = list()
logger = getLogger(__name__)

PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app")
ENV_YAML = "env.yaml"

FILE_TYPES = ["image", "audio", "video"]
TEXT_TYPES = ["text", "str", "string"]
LIST_TYPES = ["list", "tuple", "set"]
NUMBER_TYPES = ["number", "int", "integer"]
DECIMAL_TYPES = ["float", "decimal"]
BOOLEAN_TYPES = ["bool", "boolean"]
SINGULAR_TYPES = FILE_TYPES + NUMBER_TYPES + DECIMAL_TYPES + BOOLEAN_TYPES


# take several dictionaries in input and return a merged one
def merge_dicts(*args: dict) -> dict:
    """
    Merge several dictionaries into a single one.

    Args:
        *args: dictionaries to merge

    Returns:
        dict: merged dictionaries
    """

    sum_items = list()

    for dictionary in args:
        sum_items += list(dictionary.items())

    return dict(sum_items)


def dict_model(name: str, dict_def: dict) -> BaseModel:
    """
    Create a model from a dictionary.

    Args:
        name (str): name of the model
        dict_def (dict): dictionary defining the model

    Returns:
        pydantic.BaseModel: model created from the dictionary
    """
    fields = {}

    for field_name, value in dict_def.items():
        if isinstance(value, tuple):
            fields[field_name] = value
        elif isinstance(value, dict):
            fields[field_name] = (dict_model(f"{name}_{field_name}", value), ...)
        else:
            raise ValueError(f"Field {field_name}:{value} has invalid syntax")

    return create_model(name, **fields)


def get_module_infos(root_path=None) -> Tuple:
    """
    Get the list of available module infos

    Args:
        root_path (str): path to the root of the project

    Returns:
        Tuple: tuple of available module infos
    """

    if root_path:
        caller_file = root_path
    else:
        caller_file = sys._getframe(1).f_globals["__file__"]

    pwd = str(Path(caller_file).absolute()).split("/")

    plugin = pwd[len(pwd) - 3 : len(pwd)]
    tags = ".".join(plugin)
    task = plugin[-1][:-3]

    return task, plugin, tags


def get_model_versions(root_path: str = None) -> Tuple[List[str], str]:
    """
    Get the list of available model versions.

    Args:
        root_path (str): path to the root of the project

    Returns:
        Tuple: list available model versions, path to the routeur package
    """

    # used for relative paths
    if root_path:
        rel_path = root_path
    else:
        namespace = sys._getframe(1).f_globals

        cwd = os.getcwd()

        rel_path = namespace["__file__"]
        rel_path = os.path.join(cwd, rel_path)

    sub_dir_to_crawl = f"{os.path.splitext(os.path.basename(rel_path))[0]}"

    versions = dict()
    package_path = str(Path(rel_path).parent.joinpath(sub_dir_to_crawl))

    for fname in os.listdir(package_path):
        if os.path.isdir(os.path.join(package_path, fname)):
            if not Path(os.path.join(package_path, fname, "__init__.py")).exists():
                continue

            # Retieve metadata from metadata file and push it to versions,
            # the output of the get road
            model = fname
            endpoint = package_path.replace("apis", "")
            model_metadata = get_model_metadata(endpoint, model)
            versions[fname] = model_metadata

    return versions, package_path


def get_model_metadata(endpoint: str, model: str) -> dict:
    """
    Get the metadata of a model.

    Args:
        endpoint (str): name of the endpoint
        model (str): name of the model

    Returns:
        dict: metadata of the model
    """
    splited_endpoint = endpoint.split("/")
    endpoint = f"/{splited_endpoint[1]}/{splited_endpoint[2]}/{splited_endpoint[3]}/"
    path = f"apis{endpoint}{model}"
    file_name = ".model_metadata.yaml"
    fallback_file_name = ".metadata_model_template.yaml"

    return get_metadata(path, file_name, fallback_file_name)


def get_task_metadata(endpoint) -> dict:
    """
    Get the metadata of a task.

    Args:
        endpoint (str): name of the endpoint

    Returns:
        dict: metadata of the task
    """
    path = f"apis{endpoint}"
    file_name = "task.yaml"
    fallback_file_name = ".metadata_task_template.yaml"

    return get_metadata(path, file_name, fallback_file_name)


def get_metadata(rel_path, file_name, fallback_file_name) -> dict:
    """
    Get the metadata of a task or a model.

    Args:
        rel_path (str): path to the metadata file
        file_name (str): name of the metadata file
        fallback_file_name (str): name of the fallback metadata file (typicaly the template file)

    Returns:
        dict: metadata of the task or the model
    """
    file_path = os.path.join(rel_path, file_name)

    if not Path(file_path).exists():
        # make the path to fallback file fully qualified
        file_path = os.path.join(
            os.getenv("PATH_TO_GLADIA_SRC", "/app"), "apis", fallback_file_name
        )

    with open(file_path, "r") as metadata_file:
        metadata = yaml.safe_load(metadata_file)

    return metadata


def exec_in_subprocess(
    env_name: str, module_path: str, model: str, output_tmp_result: str, **kwargs
):
    """
    Execute a model in a subprocess.
    The subprocess is executed in a separate thread.

    Args:
        env_name (str): name of the environment
        module_path (str): path to the module
        model (str): name of the model to execute
        output_tmp_result (str): path to the temporary result file
        **kwargs: arguments to pass to the model

    Returns:
        threading.Thread: thread of the subprocess

    Raises:
        RuntimeError: if the subprocess fails
    """

    module_full_path = os.path.abspath(module_path)

    cwd = os.path.abspath(Path(__file__).parent)

    cmd = f"""micromamba run -n {env_name} --cwd {module_full_path} python {os.path.join(cwd, 'run_process.py')} {module_full_path} {model} {output_tmp_result} """

    cmd += f"{quote(urllib.parse.quote(json.dumps(kwargs)))}"

    logger.debug(f"Executing command: {cmd}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True,
            executable="/bin/bash",
        )

        std_outputs, error_message = proc.communicate()
        logger.debug(f"subprocess stdout: {std_outputs}")

        error_message = f"Subprocess encountered the following error : {error_message}\nCommand executed: {cmd}"

        # if the subprocess has failed (return code of shell != 0)
        # raise an exception and log to the console the error message
        if proc.returncode != 0:
            logger.error(error_message)
            raise RuntimeError(error_message)

    except subprocess.CalledProcessError as error:
        error_message = f"Could not run in subprocess command {cmd}: {error}"

        logger.error(error_message)

        raise RuntimeError(error_message)


def get_module_env_name(module_path: str) -> Union[str, None]:
    """
    Get the name of the environment from the module path.

    Args:
        module_path (str): path to the module

    Returns:
        str: name of the associated micromamba environment. None if no environment is found.
    """

    if os.path.isfile(os.path.join(module_path, ENV_YAML)):
        task_custom_env = yaml.safe_load(open(os.path.join(module_path, ENV_YAML)))

        if "name" in task_custom_env:
            return task_custom_env["name"]

        path = os.path.join(module_path, ENV_YAML).split("/")

        task = path[-3]
        model = path[-2]

        return f"{task}-{model}"

    elif os.path.isfile(os.path.join(module_path, "../", ENV_YAML)):

        model_custom_env = yaml.safe_load(
            open(os.path.join(module_path, "../", ENV_YAML))
        )

        if "name" in model_custom_env:
            return model_custom_env["name"]

        return os.path.split(os.path.split(os.path.split(module_path)[0])[0])[1]

    else:
        return None


def get_endpoint_parameter_type(parameter: dict) -> Any:
    """
    Get the type of the parameter for the endpoint and map them
    to a standard python type.

    Args:
        parameter (dict): parameter of the endpoint

    Returns:
        Any: type of the parameter for the endpoint

    Raises:
        TypeError: if the parameter type is not supported.
    """

    type_correspondence = {key: str for key in TEXT_TYPES}
    type_correspondence.update({key: int for key in NUMBER_TYPES})
    type_correspondence.update({key: Enum for key in LIST_TYPES})
    type_correspondence.update({key: float for key in DECIMAL_TYPES})
    type_correspondence.update({key: bool for key in BOOLEAN_TYPES})
    type_correspondence.update({key: Optional[UploadFile] for key in FILE_TYPES})

    parameter_type = type_correspondence.get(parameter["type"], None)

    if parameter_type == None:
        raise TypeError(f"'{parameter['type']}' is an unknown type")

    return parameter_type


def get_example_name(path: str) -> str:
    """
    Get the name of the example from the path.

    Args:
        path (str): path to the example

    Returns:
        str: name of the example
    """
    file_name_with_extension = os.path.basename(path)
    file_name, extension = os.path.splitext(file_name_with_extension)
    return f"from_{file_name}_{extension[1:]}"


def create_description_for_the_endpoint_parameter(endpoint_param: dict) -> dict:
    """
    Create a description for the endpoint parameters.
    The description is a dictionary that will be used to automatically generate
    the swagger documentation.

    Args:
        endpoint_param (dict): parameter of the endpoint

    Returns:
        dict: dict representing the description of the endpoint's parameter
    """

    parameters_to_add = {}

    parameters_to_add[endpoint_param["name"]] = {
        "type": get_endpoint_parameter_type(endpoint_param),  # i.e UploadFile
        "data_type": endpoint_param["type"],  # i.e image
        "default": None
        if endpoint_param["type"] in FILE_TYPES
        else endpoint_param.get("default", ...),
        "constructor": File if endpoint_param["type"] in FILE_TYPES else Form,
        "example": endpoint_param["example"],
        "examples": {
            get_example_name(example): example for example in endpoint_param["examples"]
        }
        if endpoint_param["type"] in FILE_TYPES and endpoint_param.get("examples", None)
        else endpoint_param.get("examples", {}),
        "description": endpoint_param.get("placeholder", ""),
    }

    # TODO: add validator checking that file and file_url can both be empty
    if endpoint_param["type"] in FILE_TYPES:
        parameters_to_add[f"{endpoint_param['name']}_url"] = {
            "type": Optional[str],
            "data_type": "url",
            "default": None,
            "constructor": Form,
            "example": endpoint_param["example"],
            "examples": {
                get_example_name(example): example
                for example in endpoint_param["examples"]
            }
            if endpoint_param.get("examples", None)
            else {},
            "description": f'{endpoint_param.get("placeholder", "")} (url) - ignored if "{endpoint_param["name"]}" file is provided',
        }

    return parameters_to_add


def get_task_examples(endpoint, models):
    task_example = dict()
    task_examples = dict()
    for model in models:
        model_metadata = get_model_metadata(endpoint, model)
        model_example = model_metadata["gladia"].get("example", {})
        model_examples = model_metadata["gladia"].get("examples", {})
        task_example.update({model: model_example})
        task_examples.update({model: model_examples})
    return task_example, task_examples


class TaskRouter:
    """
    The TaskRouter class is used to route tasks to the appropriate model.
    """

    def __init__(
        self,
        router: APIRouter,
        input: List[dict],
        output,
        default_model: str,
        rel_path=None,
    ):
        """
        Initialize the TaskRouter class
        It will create a router and all the to the main FastAPI routeur.
        It will generate 2 routes:
            - A get route that will give the list of available models and their associated metadata + a task summary
            - A post route used to "apply" the model to the input (predict endpoint) where the model is a parameter of the query
            and all other parameters are inherant to input argument from the task.py file.

        Args:
            router (APIRouter): router of the API user FastAPI engine
            input (list): a list of the input parameters of the task. input parameters are represented as dictionaries with the following keys: name, type, placeholder, example, default (if applicable)
            output (dict): a dictionary describing the output standards of the Task with the following keys: name, type, example
            default_model (str): name of the default model to use
        """
        self.input = input
        self.output = output
        self.default_model = default_model

        self.models_env = {}

        if not rel_path:
            namespace = sys._getframe(1).f_globals

            # concate the package name (i.e apis.text.text) with the model filename (i.e word-alignment.py) to obtain the relative path
            rel_path = os.path.join(
                namespace["__package__"].replace(".", "/"),
                namespace["__file__"].split("/")[-1],
            )

        full_path = os.path.join(os.getenv("PATH_TO_GLADIA_SRC", "/app"), rel_path)

        self.task_name, self.plugin, self.tags = get_module_infos(root_path=rel_path)
        self.versions, self.root_package_path = get_model_versions(full_path)
        self.endpoint = (
            f"/{rel_path.split('/')[1]}/{rel_path.split('/')[2]}/{self.task_name}/"
        )

        self.default_model = default_model

        if not self.__check_if_model_exist():
            return

        # Define the get routes implemented by fastapi
        # The @router.get() content define the informations
        # displayed in /docs and /openapi.json for the get routes
        @router.get(
            "/",
            summary=f"Get list of models available for {self.task_name}",
            tags=[self.tags],
        )
        # This function send bask the get road content to the caller
        async def get_versions():
            task_metadata = get_task_metadata(self.endpoint)
            get_content = {"models": dict(sorted(self.versions.items()))}
            # dict(sorted( is used to order
            # the models in alphabetical order
            get_content = dict(sorted(merge_dicts(get_content, task_metadata).items()))

            return get_content

        response_classes = {
            "image": ImageResponse,
            "video": VideoResponse,
            "audio": AudioResponse,
        }

        response_class = response_classes.get(self.output["type"], JSONResponse)

        response_schema = (
            response_class.schema
            if response_class in response_classes.values()
            else {
                "type": "object",
                "prediction": self.output["type"],
                "prediction_raw": Any,
            }
        )

        self.models = list(self.versions.keys())
        task_example, task_examples = get_task_examples(self.endpoint, self.models)

        responses = {
            200: {
                "content": {response_class.media_type: {"schema": response_schema}},
                "example": task_example,
                "examples": task_examples,
            }
        }

        endpoint_parameters_description = self.__build_endpoint_parameters_description(
            self.input
        )

        form_parameters = self.__build_form_parameters(endpoint_parameters_description)

        query_for_model_name = forge.arg(
            "model",
            type=str,
            default=Query(self.default_model, enum=set(self.versions.keys())),
        )

        self.inputs = self.__get_routeur_inputs()

        # Define the post routes implemented by fastapi
        # The @router.post() content define the informations
        # displayed in /docs and /openapi.json for the post routes
        @router.post(
            "/",
            summary=f"Apply model for the {self.task_name} task for a given models",
            tags=[self.tags],
            response_class=response_class,
            responses=responses,
        )
        @forge.sign(*[*form_parameters, query_for_model_name])
        async def apply(*args, **kwargs):

            # cast BaseModel pydantic models into python type
            parameters_in_body = self.__build_parameters_in_body(kwargs)
            del kwargs

            model = parameters_in_body["model"]
            del parameters_in_body["model"]

            module_path = f"{self.root_package_path}/{model}/"
            self.module_path = module_path
            if not os.path.exists(module_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {model} does not exist",
                )

            # return False if an error occured
            (
                parameters_in_body,
                success,
                error_message,
            ) = await clean_kwargs_based_on_router_inputs(
                parameters_in_body, self.inputs
            )

            if not success:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, error_message)

            if module_path not in self.models_env:
                self.models_env[module_path] = get_module_env_name(module_path)

            env_name = self.models_env[module_path]

            # if its a subprocess
            if env_name is not None:

                # check if an API is set ?
                # get the api_name to look like /input/output/task/model
                # keep the last 4 parts of the path only
                api_name = "/" + "/".join(module_path.split("/")[-5:])

                if is_subprocess_api_running(api_name):
                    # if the api is running, use it
                    result = call_subprocess_api(
                        api_name=api_name,
                        kwargs=parameters_in_body,
                    )

                else:
                    # if not exists call the subprocess
                    result = self.__call_in_subprocess(
                        inputs=self.inputs,
                        model=model,
                        kwargs=parameters_in_body,
                        env_name=env_name,
                    )

            else:
                result = self.__call_in_a_native_way(
                    model=model, kwargs=parameters_in_body, args=args
                )

            return self.__post_processing(result)

    def __get_routeur_inputs(self) -> list:
        task_metadata = yaml.safe_load(
            open(os.path.join(self.root_package_path, "task.yaml"))
        )

        return task_metadata["inputs"]

    def __build_parameters_in_body(self, kwargs: dict) -> dict:
        """
        Build the parameters in body for the post route

        Args:
            kwargs (dict): the kwargs to build the parameters in body

        Returns:
            dict: the parameters in body
        """
        parameters_in_body = dict()

        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                parameters_in_body.update(value.dict())
            else:
                parameters_in_body[key] = value

        return parameters_in_body

    def __build_endpoint_parameters_description(self, input: list) -> dict:
        """
        Build the endpoint parameters description

        Args:
            input (list): list of inputs from the router

        Returns:
            dict: the endpoint parameters description
        """
        endpoint_parameters_description = dict()
        for parameter in input:
            endpoint_parameters_description.update(
                create_description_for_the_endpoint_parameter(parameter)
            )
        return endpoint_parameters_description

    def __build_form_parameters(self, endpoint_parameters_description: dict) -> list:
        """
        Build the form parameters for the endpoint

        Args:
            endpoint_parameters_description (dict): the description of the endpoint parameters

        Returns:
            list: the list of the form parameters
        """

        form_parameters = []

        for key, value in endpoint_parameters_description.items():
            if isinstance(value["type"], EnumMeta):
                enum_values = {v: v for v in value["examples"]}
                # make the list of the enum values
                this_type = Enum(f"DynamicEnum_{time()}", enum_values)
            else:
                this_type = value["type"]

            form_parameters.append(
                forge.arg(
                    key,
                    type=this_type,
                    default=value["constructor"](
                        title=key,
                        default=value["default"],
                        description=value["description"],
                        example=value[
                            "example"
                        ],  # NOTE: FastAPI does not use this value
                        examples=value[
                            "examples"
                        ],  # NOTE: FastAPI does not use this value
                        data_type=value.get("data_type", ""),
                    ),
                )
            )
        return form_parameters

    def __check_if_model_exist(
        self,
    ) -> bool:
        """
        Verify that the default model for the task is implemented.

        Args:
            default_model (str): The default model for the task.

        Returns:
            bool: True if the default model is implemented, False otherwise.
        """

        model_dir = os.path.join(self.root_package_path, self.default_model)
        model_file = os.path.join(
            self.root_package_path, self.default_model, f"{self.default_model}.py"
        )

        if not os.path.exists(self.root_package_path):
            logger.warning(
                f"task dir ({self.root_package_path}) does not exist, skipping {self.task_name}"
            )
            return False

        elif not os.path.exists(model_dir):
            logger.warning(
                f"model_dir ({model_dir}) does not exist, skipping {self.task_name}"
            )
            return False

        elif not os.path.exists(model_file):
            logger.warning(
                f"model_file ({model_file}) does not exist, skipping {self.task_name}"
            )
            return False

        return True

    def __call_in_a_native_way(self, model: str, kwargs: dict(), args: list()) -> Any:
        """
        Call the predict function of the model in a native way.

        Args:
            model (str): name of the model
            args (list): list of args
            kwargs (dict): dict of kwargs

        Returns:
            result (any): result of the predict function
        """
        this_module = importlib.machinery.SourceFileLoader(
            model, f"{self.root_package_path}/{model}/{model}.py"
        ).load_module()

        try:
            # This is where we launch the inference without custom env
            result = getattr(this_module, f"predict")(*args, **kwargs)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"The following error occurred: {str(e)}",
            )

        return result

    def __call_in_subprocess(
        self, inputs: list, model: str, kwargs: dict, env_name: str
    ) -> Any:
        """
        Call the model in a subprocess.

        Args:
            inputs (list): list of inputs
            kwargs (dict): dict of kwargs
            env_name (str): micromamba name of the env to use

        Returns:
            Any: result of the inference ran in the subprocess
        """
        # convert io Bytes to files
        # input_files to clean
        input_files = list()
        for input_name, input_metadata in inputs.items():
            if input_metadata["type"] in FILE_TYPES:
                tmp_file = write_tmp_file(kwargs[input_name])
                kwargs[input_name] = tmp_file
                input_files.append(tmp_file)

            elif input_metadata["type"] in ["text"]:
                kwargs[input_name] = quote(kwargs[input_name])

        output_tmp_result = quote(tempfile.NamedTemporaryFile().name)

        try:
            exec_in_subprocess(
                env_name=env_name,
                module_path=self.module_path,
                model=quote(model),
                output_tmp_result=output_tmp_result,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"Error while calling the subprocess: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"The following error occurred: {str(e)}",
            )

        if is_binary_file(output_tmp_result):
            result = open(output_tmp_result, "rb").read()
        else:
            # We can't use eval due to code injection
            # (nor pickle as is use eval)
            result = json.load((open(output_tmp_result)))

        os.system(f"rm {output_tmp_result}")

        for input_file in input_files:
            os.system(f"rm {input_file}")

        return result

    def __post_processing(self, result: Any) -> Any:
        """
        Post process the result of the inference

        Args:
            result (Any): result of the inference

        Returns:
            Any
        """

        try:
            return cast_response(result, self.output)

        except Exception as e:
            error_message = f"Couldn't cast response: {e}"
            logger.error(error_message)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_message,
            )

        finally:
            if isinstance(result, str):
                try:
                    if (
                        result != "/"
                        and is_valid_path(result)
                        and os.path.exists(result)
                    ):
                        os.system(f"rm {result}")
                except Exception as e:
                    # not a valid path
                    # skip
                    logger.debug(f"not a valid path: {e}")


async def clean_kwargs_based_on_router_inputs(
    kwargs: dict, inputs: list
) -> Tuple[dict, bool, str]:
    """
    Clean the kwargs based on the router inputs.

    Args:
        kwargs (dict): dict of kwargs
        inputs (list): list of inputs

    Returns:
        Tuple[dict, bool, str]: cleaned kwargs, is_successful, error_message

    """
    success = True
    error_message = "Empty error message"

    for input_name, input_metadata in inputs.items():
        if input_metadata["type"] in FILE_TYPES:
            # if the input file is in kwargs:
            if isinstance(
                kwargs.get(input_name, None),
                starlette.datastructures.UploadFile,
            ):
                # make all io to files
                kwargs[input_name] = await kwargs[input_name].read()

            # if an url key is in the kwargs and if a file is in it
            elif kwargs.get(f"{input_name}_url", None):
                url = kwargs[f"{input_name}_url"]

                # if the url is a byte string decode it
                if isinstance(url, bytes):
                    url = url.decode("utf-8")

                dummy_header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "}

                try:
                    req = urllib.request.Request(url=url, headers=dummy_header)
                    kwargs[input_name] = urlopen(req).read()
                except (urllib.error.URLError, ValueError):
                    return ({}, False, f"Couldn't reach provided url : {url}.")

            # if not a bytes either, file is missing
            elif not isinstance(
                kwargs.get(input_name, None),
                bytes,
            ):
                error_message = f"One field among '{input_name}' and '{input_name}_url' is required."
                success = False

            # remove the url arg to avoid it to be passed in predict
            if f"{input_name}_url" in kwargs:
                del kwargs[f"{input_name}_url"]
        elif input_metadata["type"] in LIST_TYPES:
            kwargs[input_name] = str(kwargs[input_name].value)
        else:
            if not kwargs.get(input_name, None):
                error_message = f"Input '{input_name}' of '{input_metadata['type']}' type is missing."
                success = False

    return kwargs, success, error_message

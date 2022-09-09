import os
from logging import getLogger
from pathlib import Path

import yaml
from gladia_api_utils.OVHFileManager import OVHFileManager

dir_ignore = ["__pycache__"]
PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app")
logger = getLogger(__name__)


def get_all_metadata_files_path():
    """
    Get metadata files path for task and models

    Args:
        None

    Returns:
        ([task metadata path (str)], [model metadata path (str)])
    """

    model_paths = []
    task_paths = []

    # /apis/
    apis_path = os.path.join(PATH_TO_GLADIA_SRC, "apis")
    input_dir_list = os.listdir(apis_path)
    # /apis/<input>/
    for input_dir in input_dir_list:
        input_dir_path = os.path.join(apis_path, input_dir)
        if os.path.isdir(input_dir_path) and input_dir not in dir_ignore:
            output_dir_list = os.listdir(input_dir_path)
            # /apis/<input>/<output>/
            for output_dir in output_dir_list:
                output_dir_path = os.path.join(input_dir_path, output_dir)
                if os.path.isdir(output_dir_path) and output_dir not in dir_ignore:
                    task_dir_list = os.listdir(output_dir_path)
                    # /apis/<input>/<output>/<task>/
                    for task_dir in task_dir_list:
                        task_dir_path = os.path.join(output_dir_path, task_dir)
                        if os.path.isdir(task_dir_path) and task_dir not in dir_ignore:
                            model_dir_list = os.listdir(task_dir_path)
                            # /apis/<input>/<output>/<task>/<model>/
                            for model in model_dir_list:
                                model_path = os.path.join(task_dir_path, model)
                                if (
                                    os.path.isdir(model_path)
                                    and model not in dir_ignore
                                ):
                                    dir_list = os.listdir(model_path)
                                    is_model_dir = False
                                    for dir in dir_list:
                                        if dir.endswith(".py") and dir != "__init__.py":
                                            is_model_dir = True
                                            model_paths.append(model_path)
                                    if is_model_dir:
                                        task_paths.append(task_dir_path)

    return task_paths, model_paths


def update_all_metadata_fields_from_template():
    """
    Add new field from template to every metadata files.
    Args:
        None

    Returns:
        None
    """

    task_paths, model_paths = get_all_metadata_files_path()

    template_task_metadata_path = os.path.join(
        PATH_TO_GLADIA_SRC, "apis/.metadata_task_template.yaml"
    )
    template_model_metadata_path = os.path.join(
        PATH_TO_GLADIA_SRC, "apis/.metadata_model_template.yaml"
    )
    destination_model_file_name = ".model_metadata.yaml"
    destination_task_file_name = ".task_metadata.yaml"

    # Create or delete task metadata files
    for task_path in task_paths:
        task_file_path = os.path.join(task_path, destination_task_file_name)
        # Get the template metadata
        with open(template_task_metadata_path, "r") as template_metadata_file:
            source_metadata = yaml.safe_load(template_metadata_file)
        # Get the task metadata
        with open(task_file_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        # Update the task metadata from template
        for key, value in source_metadata.items():
            if key not in list(metadata.keys()):
                metadata[key] = value
        with open(task_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file)

    # Create or delete model metadata files
    for model_path in model_paths:
        model_file_path = os.path.join(model_path, destination_model_file_name)
        # Get the template metadata
        with open(template_model_metadata_path, "r") as template_metadata_file:
            source_metadata = yaml.safe_load(template_metadata_file)
        # Get the model metadata
        with open(model_file_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        # Update the model metadata from template
        for key, value in source_metadata.items():
            if key not in list(metadata.keys()):
                metadata[key] = value
        with open(model_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file)


def create_metadata_examples_with_reponse(endpoint, params, data, files, response):
    """
    Use response request to update example and examples metadata
    Args:
        endpoint (str): /<input>/<output>/<task>/
        params (dict): the params used for the request
        data (dict): the data used for the request
        files (dict): the files used for the request
        response: the request's response

    Returns:
        None
    """
    model = params["model"]
    input = endpoint.split("/")[1]
    output = endpoint.split("/")[2]

    if output != "text":

        extensions = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/png": "png",
            "audio/mpeg": "mp3",
            "audio/wav": "wav",
            "audio/x-m4a": "m4a",
        }

        fallback_extension = {
            "image": "jpg",
            "audio": "mp3",  # NOTE to change when we will have audio output model
            "video": "avi",  # NOTE to change when we will have video output model
        }

        # Upload file to OVH
        content_type = response.headers["content-type"]
        try:
            output_extension = extensions[content_type]
        except Exception as e:
            logger.error(f"Error: '{content_type}' is not a known content-type")
            output_extension = fallback_extension[output]

        if input != "text":
            file_path = (
                list(files.values())[0][1]
                if files
                else [value for key, value in data.items() if key.endswith("_url")][0]
            )
            original_file_name_with_extension = os.path.basename(file_path)
            original_file_name, input_extension = os.path.splitext(
                original_file_name_with_extension
            )
            input_extension = input_extension[1:]
            file_name = f"from_{original_file_name}_{input_extension}"
            file_name_with_extesion = f"{file_name}.{output_extension}"
            ovh_file_name = (
                f"output{endpoint}{model}/examples/{file_name_with_extesion}"
                if files
                else f"output{endpoint}{model}/example/{file_name_with_extesion}"
            )
        else:
            file_name = "output"
            file_name_with_extesion = f"{file_name}.{output_extension}"
            ovh_file_name = f"output{endpoint}{model}/example/{file_name}"

        tmp_file_path = "unit-test/tmp-output-file.png"
        with open(tmp_file_path, "wb") as out_file:
            out_file.write(response.content)
        file_manager = OVHFileManager()
        file_manager.upload_file_from_path(tmp_file_path, ovh_file_name)
        os.remove(tmp_file_path)

        # Add file link to model metadata
        metadata_file_path = get_model_metadata_path(endpoint, model)
        with open(metadata_file_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        if input != "text":
            example_dict = "examples" if files else "example"
            metadata["gladia"][example_dict][
                file_name
            ] = f"http://files.gladia.io/{ovh_file_name}"
        else:
            metadata["gladia"]["example"][
                "output"
            ] = f"http://files.gladia.io/{ovh_file_name}"
        with open(metadata_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file)

    else:
        metadata_file_path = get_model_metadata_path(endpoint, model)
        with open(metadata_file_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        metadata["gladia"]["example"]["output"] = response.json()
        with open(metadata_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file)


def get_model_metadata_path(endpoint, model):
    """
    Retieve the path of the model metadata file.

    Args:
        endpoint (str): /<input>/<output>/<task>/
        model (str): model name
    Returns:
        model metadata file path (str)
    """
    task_name = endpoint.split("/")[3]
    task_models_folder = endpoint.replace(task_name, f"{task_name}-models")
    source_folder = PATH_TO_GLADIA_SRC
    path_to_metadata = f"{source_folder}/apis{task_models_folder}{model}"
    metadata_file_name = ".model_metadata.yaml"
    metadata_file_path = os.path.join(path_to_metadata, metadata_file_name)
    if not Path(metadata_file_path).exists():
        with open(
            f"{source_folder}/apis/.metadata_model_template.yaml", "r"
        ) as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        with open(metadata_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file)
    return metadata_file_path


def clean_up_model_output_data(endpoint, model):
    """
    Delete the outputs saved in File Storage and metadata for a given model.

    Args:
        endpoint (str): /<input>/<output>/<task>/
        model (str): model name
    Returns:
        None
    """
    # Clean up files in OVH Object Storage
    file_manager = OVHFileManager()
    prefix = f"output{endpoint}{model}"
    files_to_delete = file_manager.get_objects(prefix=prefix)
    for file_to_delete in files_to_delete:
        file_manager.delete_file(file_to_delete)
    # Clean up metadata
    metadata_file_path = get_model_metadata_path(endpoint, model)
    with open(metadata_file_path, "r") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    metadata["gladia"]["example"] = {}
    metadata["gladia"]["examples"] = {}
    with open(metadata_file_path, "w") as metadata_file:
        yaml.dump(metadata, metadata_file)

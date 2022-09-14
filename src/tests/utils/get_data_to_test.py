import os
import yaml

from typing import Any, Dict, List
from _pytest.config import _prepareconfig
from itertools import product


PYTEST_CONFIG = None


def get_models_to_test(path_to_task: str) -> List[str]:
    """
    Get a list of models to test for a given task. The list is obtained by
    looking for all the directories in the task directory that contain a
    `model.py` file.

    If the command line argument --default-models-only is set, only the default models are returned.

    Args:
        path_to_task (str): Path to the task directory.

    Returns:
        List[str]: List of models to test.
    """

    global PYTEST_CONFIG

    # prevents overwriting gunicorn's command line parser
    if "gunicorn" not in os.environ["_"]:

        if PYTEST_CONFIG is None:
            PYTEST_CONFIG = _prepareconfig()

        if PYTEST_CONFIG.getoption("--default-models-only"):
            return set([""])

    models = [model for model in os.listdir(path_to_task) if model[0] not in [".", "_"]]
    models = [
        model for model in models if os.path.isdir(os.path.join(path_to_task, model))
    ]
    models = [
        model
        for model in models
        if os.path.isfile(os.path.join(path_to_task, model, f"{model}.py"))
    ]

    return set(models)


def get_inputs_to_test(path_to_task: str, input_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve the test values for each input specified in `input_names`

    Args:
        path_to_task (str): Path to the task directory.
        input_names (List[str]): list of inputs to retrieve test values for.

    Returns:
        List[Dict[str, Any]]: List of combinations of values to test
    """

    global PYTEST_CONFIG

    # prevents overwriting gunicorn's command line parser
    if "gunicorn" not in os.environ["_"]:

        if PYTEST_CONFIG is None:
            PYTEST_CONFIG = _prepareconfig()

    else:
        return [{input_name: "" for input_name in input_names}]

    task_metadata_file_path = os.path.join(path_to_task, ".task_metadata.yaml")
    task_metadata = yaml.safe_load(open(task_metadata_file_path, "r"))

    if PYTEST_CONFIG.getoption("--default-inputs-only"):
        return [{input_name : task_metadata["inputs_example"][input_name]["default_example"] for input_name in input_names}]

    possible_values_for_each_input = {input_name : task_metadata["inputs_example"][input_name]["examples"] for input_name in input_names}

    return [dict(zip(possible_values_for_each_input, v)) for v in product(*possible_values_for_each_input.values())]

import os
import pathlib
import sys
from itertools import product
from typing import Any, Dict, List

import yaml
from _pytest.config import _prepareconfig

PYTEST_CONFIG = None


# TODO: default value for path_to_task


def get_models_to_test(path_to_task: str = None) -> List[str]:
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

    if path_to_task is None:
        path_to_task = str(
            pathlib.Path(sys._getframe(1).f_globals["__file__"]).parents[0].absolute()
        )

    global PYTEST_CONFIG

    # prevents overwriting gunicorn's command line parser
    if "gunicorn" not in os.environ.get("_", "gunicorn"):

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


def get_inputs_to_test(
    input_names: List[str], path_to_task: str = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the test values for each input specified in `input_names`

    Args:
        path_to_task (str): Path to the task directory.
        input_names (List[str]): list of inputs to retrieve test values for.

    Returns:
        List[Dict[str, Any]]: List of combinations of values to test
    """

    if path_to_task is None:
        path_to_task = str(
            pathlib.Path(sys._getframe(1).f_globals["__file__"]).parents[0].absolute()
        )

    global PYTEST_CONFIG

    # prevents overwriting gunicorn's command line parser
    if "gunicorn" not in os.environ.get("_", "gunicorn"):

        if PYTEST_CONFIG is None:
            PYTEST_CONFIG = _prepareconfig()

    else:
        return [{input_name: "" for input_name in input_names}]

    task_metadata_file_path = os.path.join(path_to_task, "task.yaml")
    task_metadata = yaml.safe_load(open(task_metadata_file_path, "r"))

    if PYTEST_CONFIG.getoption("--default-inputs-only"):
        return [
            {
                input_name: task_metadata["inputs"][input_name.replace("_url", "")][
                    "examples"
                ][0]
                for input_name in input_names
            }
        ]

    possible_values_for_each_input = {
        input_name: task_metadata["inputs"][input_name.replace("_url", "")]["examples"]
        for input_name in input_names
    }

    return [
        dict(zip(possible_values_for_each_input, v))
        for v in product(*possible_values_for_each_input.values())
    ]

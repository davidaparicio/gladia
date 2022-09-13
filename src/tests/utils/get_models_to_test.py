import os
from typing import List

from _pytest import config as __pytest_config

PYTEST_CONFIG = __pytest_config._prepareconfig()


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

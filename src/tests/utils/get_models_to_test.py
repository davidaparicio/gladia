import os
from typing import List

from _pytest.config import _prepareconfig

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

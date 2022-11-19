import os
from typing import Any

import yaml


def get_task_metadata(path_to_initializer_file: str) -> Any:
    """
    Retrieve the task metadata for a task

    Args:
        path_to_initializer_file (str): file initializing the task (apis/{input}/{output}/{task.py})

    Returns:
        Any: content of the task.yaml
    """

    task_metadata_file_path = os.path.join(
        os.path.split(path_to_initializer_file)[0],
        os.path.split(path_to_initializer_file)[1],
        "task.yaml",
    )

    return yaml.safe_load(open(task_metadata_file_path, "r"))

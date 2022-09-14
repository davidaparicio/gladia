import os
from typing import Any

import yaml


def get_task_metadata(initializer_file: str) -> Any:
    """
    Retrieve the task metadata for a task

    Args:
        initializer_file (str): file initializing the task (apis/{input}/{output}/{task.py})

    Returns:
        Any: content of the .task_metadata.yaml
    """

    task_metadata_file_path = os.path.join(
        os.path.split(initializer_file)[0],
        os.path.split(initializer_file)[1].replace(".py", "-models"),
        ".task_metadata.yaml",
    )

    return yaml.safe_load(open(task_metadata_file_path, "r"))

import os
from typing import List


def get_models_to_test(path_to_task) -> List[str]:

    if os.getenv("TEST_DEFAULT_MODELS_ONLY", None):
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

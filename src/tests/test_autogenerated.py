import json
import os
from logging import getLogger
from os.path import join as join_path

import pytest
import requests
from gladia_api_utils.task_management import get_task_metadata

from main import app
from tests.default_tests import create_default_tests
from tests.utils import get_inputs_to_test, get_models_to_test

logger = getLogger(__name__)


def __a_model_exists_for_this_task(
    path_to_api, input_modality, output_modality, task
) -> bool:
    files_and_files = os.listdir(
        join_path(path_to_api, input_modality, output_modality, task)
    )
    models = list(
        filter(
            lambda f: f[0] not in ("_", ".")
            and os.path.isdir(
                join_path(path_to_api, input_modality, output_modality, task, f)
            ),
            files_and_files,
        )
    )

    if len(models) == 0:
        return False

    return True


def task_is_activated(
    activated_tasks, path_to_api, input_modality, output_modality, task_dir
) -> bool:
    task = task_dir.replace("-models", "")

    if "*" in activated_tasks[input_modality] and __a_model_exists_for_this_task(
        path_to_api, input_modality, output_modality, task_dir
    ):
        return True

    if output_modality not in activated_tasks[input_modality]:
        return False

    if (
        "*" in activated_tasks[input_modality][output_modality]
        or task.replace("-models", "")
        in activated_tasks[input_modality][output_modality]
    ) and __a_model_exists_for_this_task(
        path_to_api, input_modality, output_modality, task_dir
    ):
        return True

    return False


def autogenerate_tests(path_to_api: str, path_to_config: str):
    import requests

    activated_tasks = json.load(open(path_to_config))["active_tasks"]

    tests = []

    for input_modality in os.listdir(path_to_api):

        if os.path.isfile(join_path(path_to_api, input_modality)):
            continue

        for output_modality in os.listdir(join_path(path_to_api, input_modality)):

            if os.path.isfile(join_path(path_to_api, input_modality, output_modality)):
                continue

            for task in os.listdir(
                join_path(path_to_api, input_modality, output_modality)
            ):

                if task.endswith("-models") is False:
                    continue

                if (
                    task_is_activated(
                        activated_tasks,
                        path_to_api,
                        input_modality,
                        output_modality,
                        task,
                    )
                    is False
                ):
                    continue

                path_to_initializer_file = join_path(
                    path_to_api,
                    input_modality,
                    output_modality,
                    f"{task[:-len('-models')]}.py",
                )

                if os.path.exists(path_to_initializer_file) is False:
                    continue

                task_metadata = get_task_metadata(
                    path_to_initializer_file=path_to_initializer_file
                )

                input_names = []

                for input_name in list(task_metadata["inputs"].keys()):
                    if task_metadata["inputs"][input_name]["type"] in (
                        "audio",
                        "video",
                        "image",
                    ):
                        input_names.append(input_name + "_url")
                    else:
                        input_names.append(input_name)

                inputs_to_test = get_inputs_to_test(
                    input_names=input_names,
                    path_to_task=join_path(
                        path_to_api, input_modality, output_modality, task
                    ),
                )

                tests.append(
                    pytest.mark.autogenerated()(
                        create_default_tests(
                            class_name=f"{input_modality.capitalize()}{output_modality.capitalize()}{task.replace('-models', '').replace('-', '').capitalize()}",
                            client=requests,
                            target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8080'))}/{input_modality}/{output_modality}/{task.replace('-models', '')}/",
                            models_to_test=get_models_to_test(
                                path_to_task=join_path(
                                    path_to_api, input_modality, output_modality, task
                                )
                            ),
                            inputs_to_test=inputs_to_test,
                        )
                    )
                )

    return tests


TestAutogenerated = type(
    "TestAutogenerated",
    (()),
    {
        f"Test{test.__name__}": test
        for test in autogenerate_tests(
            path_to_api=join_path(os.getenv("PATH_TO_GLADIA_SRC"), "apis"),
            path_to_config=join_path(os.getenv("PATH_TO_GLADIA_SRC"), "config.json"),
        )
    },
)

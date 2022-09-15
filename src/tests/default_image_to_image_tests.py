import os
import pytest
import tempfile
import requests

from typing import Any, Dict, List
from fastapi.testclient import TestClient
from .constants import HOST_TO_EXAMPLE_STORAGE


def __apply_decorators(func, *decorators):

    def deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return deco(func)


def __test_image_input_task(self, model: str, inputs: Dict[str, Any]) -> bool:
    """
    Test the endpoint with a jpg image input

    Args:
        model (str): model to test
        inputs (Dict[str, Any]): input values to test

    Returns:
        bool: True if the test passed, False otherwise
    """

    tmp_original_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    tmp_original_image_file.write(
        requests.get(inputs["image_url"]).content
    )

    response = self.client.post(
        url=self.target_url,
        params={"model": model} if model else {},
        files={
            "image": open(tmp_original_image_file.name, "rb"),
        },
    )

    assert response.status_code == 200


def __test_image_url_input_task(self, model: str, inputs: Dict[str, Any]) -> bool:
    """
    Test the endpoint with a jpg image input retrieved from an url

    Args:
        model (str): model to test
        inputs (Dict[str, Any]): input values to test

    Returns:
        bool: True if the test passed, False otherwise
    """

    response = self.client.post(
        url=self.target_url,
        params={"model": model} if model else {},
        data={
            "image_url": inputs["image_url"],
        },
    )

    assert response.status_code == 200


def __test_invalid_image_input_task(self, model: str) -> bool:
    """
    Test the endpoint with an invalid image input

    Args:
        model (str): model to test

    Returns:
        bool: True if the test passed, False otherwise
    """

    tmp_local_mp3_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    tmp_local_mp3_file.write(
        requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content
    )

    with pytest.raises(Exception):
        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "image": (
                    f"{tmp_local_mp3_file.name}.mp3",
                    open(tmp_local_mp3_file.name, "rb"),
                    "audio/mpeg",
                )
            },
        )

        # assert response.status_code != 200 # TODO

    tmp_local_mp3_file.close()
    os.unlink(tmp_local_mp3_file.name)


def __test_invalid_image_url_input_task(self, model: str) -> bool:
    """
    Test the endpoint with an invalid image url input

    Args:
        model (str): model to test

    Returns:
        bool: True if the test passed, False otherwise
    """

    with pytest.raises(Exception):
        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={"image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4"},
        )

        # assert response.status_code != 200 # TODO


def __test_empty_input_task(self, model: str) -> bool:
    """
    Test the endpoint with an empty input

    Args:
        model (str): model to test

    Returns:
        bool: True if the test passed, False otherwise
    """
    response = self.client.post(
        url=self.target_url,
        params={"model": model} if model else {},
        data={},
    )

    assert response.status_code == 200  # TODO: change to != 200


# TODO: inherit from IBasicTest
def create_default_image_to_image_tests(
    class_name: str,
    client: TestClient,
    target_url: str,
    models_to_test: List[str],
    inputs_to_test: List[Dict[str, Any]]
):
    return type(
        class_name, (),
        {
            "client": client,
            "target_url": target_url,
            "test_image_input_task": __apply_decorators(
                __test_image_input_task,
                pytest.mark.mandatory(),
                pytest.mark.parametrize("model", models_to_test),
                pytest.mark.parametrize("inputs", inputs_to_test)
            ),
            "test_image_url_input_task": __apply_decorators(
                __test_image_url_input_task,
                pytest.mark.mandatory(),
                pytest.mark.parametrize("model", models_to_test),
                pytest.mark.parametrize("inputs", inputs_to_test)
            ),
            "test_invalid_image_input_task": __apply_decorators(
                __test_invalid_image_input_task,
                pytest.mark.parametrize("model", models_to_test),
            ),
            "test_invalid_image_url_input_task": __apply_decorators(
                __test_invalid_image_url_input_task,
                pytest.mark.parametrize("model", models_to_test),
            ),
            "test_empty_input_task": __apply_decorators(
                __test_empty_input_task,
                pytest.mark.parametrize("model", models_to_test),
            ),
        },
    )
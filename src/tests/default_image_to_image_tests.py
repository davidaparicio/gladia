import os
import tempfile
from typing import Any, Callable, Dict, List

import pytest
import requests
from fastapi.testclient import TestClient

from .constants import HOST_TO_EXAMPLE_STORAGE


def __apply_decorators(func, *decorators):
    def deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return deco(func)


def get_test_image_input_task(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

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
        tmp_original_image_file.write(requests.get(inputs["image_url"]).content)

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "image": open(tmp_original_image_file.name, "rb"),
            },
        )

        assert (
            response.status_code == 200
        ), f"client returned a non 200 status code: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_image_input_task,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_image_url_input_task(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image_url inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

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

        assert (
            response.status_code == 200
        ), f"client returned a non 200 status code: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_image_url_input_task,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_invalid_image_input_task(
    models_to_test: List[str],
) -> Callable[[str], bool]:
    """
    Generate the test function testing the endpoint with an invalid image input

    Args:
        models_to_test (List[str]): models to test

    Returns:
        Callable[[str], bool]: test function
    """

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

        assert (
            response.status_code == 500
        ), f"client did not returned a 500 status code: {response.status_code}. Client's response: {response.content}"

        tmp_local_mp3_file.close()
        os.unlink(tmp_local_mp3_file.name)

    return __apply_decorators(
        __test_invalid_image_input_task,
        pytest.mark.parametrize("model", models_to_test),
    )


def get_test_invalid_image_url_input_task(
    models_to_test: List[str],
) -> Callable[[str], bool]:
    """
    Generate the test function testing the endpoint with an invalid image url input

    Args:
        models_to_test (List[str]): models to test

    Returns:
        Callable[[str], bool]: test function
    """

    def __test_invalid_image_url_input_task(self, model: str) -> bool:
        """
        Test the endpoint with an invalid image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={"image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4"},
        )

        assert (
            response.status_code == 500
        ), f"client did not returned a 500 status code: {response.status_code}. Client's response: {response.content}"

    return __apply_decorators(
        __test_invalid_image_url_input_task,
        pytest.mark.parametrize("model", models_to_test),
    )


def get_test_empty_input_task(models_to_test: List[str]) -> Callable[[str], bool]:
    """
    Generate the test function testing the endpoint with an empty input

    Args:
        models_to_test (List[str]): models to test

    Returns:
        Callable[[str], bool]: test function
    """

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

    return __apply_decorators(
        __test_empty_input_task,
        pytest.mark.parametrize("model", models_to_test),
    )


def create_default_image_to_image_tests(
    class_name: str,
    client: TestClient,
    target_url: str,
    models_to_test: List[str],
    inputs_to_test: List[Dict[str, Any]],
):
    return type(
        class_name,
        (),
        {
            "client": client,
            "target_url": target_url,
            "test_image_input_task": get_test_image_input_task(
                models_to_test, inputs_to_test
            ),
            "test_image_url_input_task": get_test_image_url_input_task(
                models_to_test, inputs_to_test
            ),
            "test_invalid_image_input_task": get_test_invalid_image_input_task(
                models_to_test
            ),
            "test_invalid_image_url_input_task": get_test_invalid_image_url_input_task(
                models_to_test
            ),
            "test_empty_input_task": get_test_empty_input_task(models_to_test),
        },
    )

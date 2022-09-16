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



def get_basic_input_test(
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

    def __test_basic_input(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the endpoint with a valid inputs

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data=inputs,
        )

        assert (
            response.status_code == 200
        ), f"client returned a non 200 status code: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_basic_input,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_invalid_input_test(
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

    def __test_invalid_input(self, model: str, inputs: Dict[str, Any] = inputs_to_test[0]) -> bool:
        """
        Test the endpoint with a valid inputs

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={key: bytearray(b'\x00\x0F') for (key, _) in inputs.items()},
        )

        assert (
            response.status_code == 500
        ), f"expected 500 status code bu received: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_invalid_input,
        pytest.mark.parametrize("model", models_to_test),
    )


def get_test_empty_input(models_to_test: List[str]) -> Callable[[str], bool]:
    """
    Generate the test function testing the endpoint with an empty input

    Args:
        models_to_test (List[str]): models to test

    Returns:
        Callable[[str], bool]: test function
    """

    def __test_empty_input(self, model: str) -> bool:
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

        assert response.status_code == 422, f"expected 422 but received {response.status_code}"  # TODO: change to != 200

    return __apply_decorators(
        __test_empty_input,
        pytest.mark.parametrize("model", models_to_test),
    )


def create_default_text_to_text_tests(
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
            "test_basic_input": get_basic_input_test(models_to_test, inputs_to_test),
            # "test_invalid_input": get_invalid_input_test(models_to_test, inputs_to_test), # TODO
            "test_empty_input": get_test_empty_input(models_to_test),
        },
    )

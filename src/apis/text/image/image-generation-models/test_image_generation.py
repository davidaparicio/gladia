import os
from typing import Any, Dict, List

import pytest
import requests

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["prompt", "samples", "steps", "seed"])


class TestImageGeneration:
    """
    Class to test the image generation endpoint
    """

    target_url = f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/text/image/image-generation/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_basic_input(self, model: str, inputs: List[Dict[str, Any]]) -> bool:
        """
        Test the image generation endpoint with basic inputs

        Args:
            model (str): model to test
            inputs (List[Dict[str, Any]]): inputs to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "prompt": inputs["prompt"],
                "samples": inputs["samples"],
                "steps": inputs["steps"],
                "seed": inputs["seed"],
            },
        )

        assert (
            response.status_code == 200
        ), f"non-200 status code: {response.status_code} with the following message: {response.content}"

    @pytest.mark.parametrize("model", models)
    def test_invalid_prompt_input(self, model: str) -> bool:
        """
        Test the image generation endpoint with basic inputs

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "prompt": b"0101010101",
                "samples": "some random value",
                "steps": inputs_to_test[0]["steps"],
                "seed": inputs_to_test[0]["seed"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_samples_input(self, model: str) -> bool:
        """
        Test the image generation endpoint with basic inputs

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "prompt": inputs_to_test[0]["prompt"],
                "samples": "some random value",
                "steps": inputs_to_test[0]["steps"],
                "seed": inputs_to_test[0]["seed"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_steps_input(self, model: str) -> bool:
        """
        Test the image generation endpoint with basic inputs

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "prompt": inputs_to_test[0]["prompt"],
                "samples": inputs_to_test[0]["samples"],
                "steps": "some random value",
                "seed": inputs_to_test[0]["seed"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_seed_input(self, model: str) -> bool:
        """
        Test the image generation endpoint with basic inputs

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "prompt": inputs_to_test[0]["prompt"],
                "samples": inputs_to_test[0]["samples"],
                "steps": inputs_to_test[0]["steps"],
                "seed": "some random value",
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_url_input_task(self, model: str) -> bool:
        """
        Test the image generation endpoint with an invalid original image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={"image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4"},
        )

        assert response.status_code == 422

    # @pytest.mark.skip("Model neither crash nor return a non-200 status code") #FIXME
    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the image generation endpoint with an empty input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={},
        )

        assert response.status_code == 422

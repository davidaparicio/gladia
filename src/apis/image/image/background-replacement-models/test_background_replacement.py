import os
import tempfile
from typing import Any, Dict

import pytest
import requests
from fastapi.testclient import TestClient

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE
from tests.utils import get_inputs_to_test, get_models_to_test

client = TestClient(app)

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["original_image_url", "background_image_url", "alignment"])


class TestGuidedInpainting:
    """
    Class to test the guided inpainting endpoint
    """

    target_url = "/image/image/background-replacement/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_local_inputs_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the background replacement endpoint with a image inputs

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": requests.get(inputs["original_image_url"]).content,
                "background_image": requests.get(inputs["background_image_url"]).content,
            },
            data={
                "alignment": inputs["alignment"],
            },
        )

        assert response.status_code == 200

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_url_input_task(self, model: str, inputs) -> bool:
        """
        Test the background replacement endpoint with a image inputs retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": inputs["original_image_url"],
                "background_image_url": inputs["background_image_url"],
                "alignment": inputs["alignment"],
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the background replacement endpoint with an invalid original image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": (
                    "test.mp3",
                    requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content,
                    "audio/mpeg",
                ),
                "background_image_url": inputs_to_test[0]["background_image_url"],
            },
            data={
                "prompt": inputs_to_test[0]["prompt"],
            },
        )

        assert response.status_code == 500  # TODO

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the background replacement endpoint with an invalid background image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image_url": inputs_to_test[0]["original_image_url"],
                "background_image": (
                    "test.mp3",
                    requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content,
                    "audio/mpeg",
                ),
            },
            data={
                "alignment": inputs_to_test[0]["alignment"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_url_input_task(self, model: str) -> bool:
        """
        Test the background replacement endpoint with an invalid original image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                "background_image_url": inputs_to_test[0]["background_image_url"],
                "alignment": inputs_to_test[0]["alignment"],
            },
        )

        assert response.status_code == 500

    @pytest.mark.skip  # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_mask_image_url_input_task(self, model: str) -> bool:
        """
        Test the background replacement endpoint with an invalid background image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": inputs_to_test[0]["original_image_url"],
                "background_image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                "alignment": inputs_to_test[0]["alignment"],
            },
        )

        assert response.status_code == 500

    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the guided inpainting endpoint with an empty input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={},
        )

        assert response.status_code == 422

import os
from typing import Any, Dict

import pytest
import requests

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["original_image_url", "mask_image_url", "prompt"])


class TestGuidedInpainting:
    """
    Class to test the guided inpainting endpoint
    """

    target_url = f"http://{os.getenv('TEST_CLIENT_HOST', '0.0.0.0')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/image/image/guided-inpainting/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_local_inputs_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the guided inpainting endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": requests.get(inputs["original_image_url"]).content,
                "mask_image": requests.get(inputs["mask_image_url"]).content,
            },
            data={
                "prompt": inputs["prompt"],
            },
        )

        assert (
            response.status_code == 200
        ), f"expected 200 but received {response.status_code}, body: {response.content}"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_url_input_task(self, model: str, inputs) -> bool:
        """
        Test the guided inpainting endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": inputs["original_image_url"],
                "mask_image_url": inputs["mask_image_url"],
                "prompt": inputs["prompt"],
            },
        )

        assert (
            response.status_code == 200
        ), f"expected 200 but received {response.status_code}, body: {response.content}"

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str, tmp_path) -> bool:
        """
        Test the guided inpainting endpoint with an invalid original image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": (
                    "test.mp3",
                    requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content,
                    "audio/mpeg",
                ),
                "mask_image_url": inputs_to_test[0]["mask_image_url"],
            },
            data={
                "prompt": inputs_to_test[0]["prompt"],
            },
        )

        assert response.status_code == 500  # TODO

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the guided inpainting endpoint with an invalid mask image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image_url": inputs_to_test[0]["original_image_url"],
                "mask_image": (
                    "test.mp3",
                    requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content,
                    "audio/mpeg",
                ),
            },
            data={
                "prompt": inputs_to_test[0]["prompt"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_url_input_task(self, model: str) -> bool:
        """
        Test the guided inpainting endpoint with an invalid original image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                "mask_image_url": inputs_to_test[0]["mask_image_url"],
                "prompt": inputs_to_test[0]["prompt"],
            },
        )

        assert response.status_code == 500

    @pytest.mark.skip  # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_mask_image_url_input_task(self, model: str) -> bool:
        """
        Test the guided inpainting endpoint with an invalid mask image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": inputs_to_test[0]["original_image_url"],
                "mask_image": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                "prompt": inputs_to_test[0]["prompt"],
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

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={},
        )

        assert response.status_code == 422

import os
from typing import Any, Dict

import pytest
import requests

from tester.constants import HOST_TO_EXAMPLE_STORAGE
from tester.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["original_image_url", "mask_image_url"])


class TestInpainting:
    """
    Class to test the inpainting endpoint
    """

    target_url = f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8080'))}/image/image/inpainting/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_local_inputs_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the inpainting endpoint with a jpg image input

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
        )

        assert (
            response.status_code == 200
        ), f"expected 200 but received {response.status_code}, body: {response.content}"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_url_input_task(self, model: str, inputs) -> bool:
        """
        Test the inpainting endpoint with a jpg image input retrieved from an url

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
            },
        )

        assert (
            response.status_code == 200
        ), f"expected 200 but received {response.status_code}, body: {response.content}"

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid original image input

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
        )

        assert response.status_code == 500

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid mask image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": inputs_to_test[0]["original_image_url"],
                "mask_image": (
                    "test.mp3",
                    requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content,
                    "audio/mpeg",
                ),
            },
        )

        assert response.status_code == 500

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_url_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid original image url input

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
            },
        )

        assert response.status_code == 500  # TODO

    @pytest.mark.skip  # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_mask_image_url_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid mask image url input

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
            },
        )

        assert response.status_code == 500

    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an empty input

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

        assert (
            response.status_code == 200
        ), f"expected 200 but received {response.status_code}, body: {response.content}"  # TODO: change to != 200
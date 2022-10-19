import os
import tempfile
from typing import Any, Dict

import pytest
import requests

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["image_url", "source_language"])


class TestOcr:
    """
    Class to test the ocr endpoint
    """

    target_url = f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/image/text/ocr/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_local_inputs_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the ocr endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        tmp_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_image_file.write(requests.get(inputs["image_url"]).content)

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "image": open(tmp_image_file.name, "rb"),
            },
            data={
                "source_language": inputs["source_language"],
            },
        )

        assert (
            response.status_code == 200
        ), f"non-200 status code: {response.status_code} error message: {response.content}"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_url_input_task(self, model: str, inputs) -> bool:
        """
        Test the ocr endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "image_url": inputs["image_url"],
                "source_language": inputs["source_language"],
            },
        )

        assert (
            response.status_code == 200
        ), f"non-200 status code: {response.status_code} with the following message: {response.content}"

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_input_task(self, model: str) -> bool:
        """
        Test the ocr endpoint with an invalid mask image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        tmp_local_mp3_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_local_mp3_file.write(
            requests.get(f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp3").content
        )

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "image": (
                    tmp_local_mp3_file.name,
                    open(tmp_local_mp3_file.name, "rb"),
                    "audio/mpeg",
                ),
            },
            data={
                "source_language": inputs_to_test[0]["source_language"],
            },
        )

        tmp_local_mp3_file.close()
        os.unlink(tmp_local_mp3_file.name)

        assert response.status_code == 500

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_url_input_task(self, model: str) -> bool:
        """
        Test the ocr endpoint with an invalid original image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = requests.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                "source_language": inputs_to_test[0]["source_language"],
            },
        )

        assert response.status_code == 500

    @pytest.mark.skip("Model neither crash nor return a non-200 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the ocr endpoint with an empty input

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

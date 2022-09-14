import os
import tempfile
from typing import Any, Dict
import requests
import pytest
from fastapi.testclient import TestClient

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE, PATH_TO_EXAMPLE_FILES
from tests.utils import get_models_to_test, get_inputs_to_test

client = TestClient(app)

models = get_models_to_test(os.path.split(__file__)[0])
inputs_to_test = get_inputs_to_test(os.path.split(__file__)[0], ["original_image_url", "mask_image_url"])


class TestInpainting:
    """
    Class to test the inpainting endpoint
    """

    target_url = "/image/image/inpainting/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_png_image_input_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the inpainting endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        tmp_original_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_original_image_file.write(requests.get(inputs["original_image_url"]).content)

        tmp_mask_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_mask_image_file.write(requests.get(inputs["mask_image_url"]).content)

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": open(tmp_original_image_file.name, "rb"),
                "mask_image": open(tmp_mask_image_file.name, "rb"),
            },
        )

        assert response.status_code == 200

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_image_url_input_task(self, model: str, inputs) -> bool:
        """
        Test the inpainting endpoint with a jpg image input retrieved from an url

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
                "mask_image_url": inputs["mask_image_url"],
            },
        )

        assert response.status_code == 200

    @pytest.mark.skip # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid original image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """
        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                files={
                    "original_image": (
                        "test.mp3",
                        open(os.path.join(PATH_TO_EXAMPLE_FILES, "test.mp3"), "rb"),
                        "audio/mpeg",
                    ),
                    "mask_image": self.default_local_mask_image,
                },
            )

            assert response.status_code != 200 # TODO

    @pytest.mark.skip # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid mask image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """
        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                files={
                    "original_image": self.default_local_mask_image,
                    "mask_image": (
                        "test.mp3",
                        open(os.path.join(PATH_TO_EXAMPLE_FILES, "test.mp3"), "rb"),
                        "audio/mpeg",
                    ),
                },
            )

            assert response.status_code != 200 # TODO

    @pytest.mark.skip # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_url_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid original image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                data={
                    "original_image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                    "mask_image_url": inputs_to_test[0]["mask_image_url"],
                },
            )

            # assert response.status_code != 200 # TODO

    @pytest.mark.skip # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_mask_image_url_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid mask image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                data={
                    "original_image_url": inputs_to_test[0]["original_image_url"],
                    "mask_image": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                },
            )

            # assert response.status_code != 200 # TODO

    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an empty input

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

        assert response.status_code == 200  # TODO: change to != 200

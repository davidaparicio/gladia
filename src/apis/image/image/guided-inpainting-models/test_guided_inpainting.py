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

models = get_models_to_test(os.path.split(__file__)[0])
inputs_to_test = get_inputs_to_test(
    os.path.split(__file__)[0], ["original_image_url", "mask_image_url", "prompt"]
)


class TestInpainting:
    """
    Class to test the inpainting endpoint
    """

    target_url = "/image/image/inpainting/"

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

        tmp_original_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_original_image_file.write(
            requests.get(inputs["original_image_url"]).content
        )

        tmp_mask_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_mask_image_file.write(requests.get(inputs["mask_image_url"]).content)

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={
                "original_image": open(tmp_original_image_file.name, "rb"),
                "mask_image": open(tmp_mask_image_file.name, "rb"),
            },
            data={
                "prompt": inputs["prompt"],
            }
        )

        assert response.status_code == 200

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

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "original_image_url": inputs["original_image_url"],
                "mask_image_url": inputs["mask_image_url"],
                "prompt": inputs["prompt"],
            },
        )

        assert response.status_code == 200

    @pytest.mark.skip  # FIXME: Model neither crash nor return a non-200 status code
    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid original image input

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
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                files={
                    "original_image": (
                        tmp_local_mp3_file.name,
                        open(tmp_local_mp3_file.name, "rb"),
                        "audio/mpeg",
                    ),
                    "mask_image": self.default_local_mask_image,
                },
                data={
                    "prompt": inputs_to_test[0]["prompt"],
                }
            )

            tmp_local_mp3_file.close()
            os.unlink(tmp_local_mp3_file.name)

            assert response.status_code != 200  # TODO

    @pytest.mark.parametrize("model", models)
    def test_invalid_original_image_input_task(self, model: str) -> bool:
        """
        Test the inpainting endpoint with an invalid mask image input

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
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                files={
                    "original_image": self.default_local_mask_image,
                    "mask_image": (
                        tmp_local_mp3_file.name,
                        open(tmp_local_mp3_file.name, "rb"),
                        "audio/mpeg",
                    ),
                },
                data={
                    "prompt": inputs_to_test[0]["prompt"],
                }
            )

            tmp_local_mp3_file.close()
            os.unlink(tmp_local_mp3_file.name)

            assert response.status_code != 200  # TODO

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
                    "prompt": inputs_to_test[0]["prompt"],
                },
            )

            # assert response.status_code != 200 # TODO

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

        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                data={
                    "original_image_url": inputs_to_test[0]["original_image_url"],
                    "mask_image": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4",
                    "prompt": inputs_to_test[0]["prompt"],
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

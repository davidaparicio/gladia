import os
import tempfile
from typing import Any, Dict
from urllib.request import urlretrieve

import pytest
import requests
from fastapi.testclient import TestClient

from apis.image.image.IBasicTestsImageToImage import IBasicTestsImageToImage
from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE
from tests.utils import get_inputs_to_test, get_models_to_test

client = TestClient(app)

models = get_models_to_test(os.path.split(__file__)[0])
inputs_to_test = get_inputs_to_test(os.path.split(__file__)[0], ["image_url"])


class TestBackgroundRemoval(IBasicTestsImageToImage):
    """
    Class to test the background removal endpoint
    """

    target_url = "/image/image/background-removal/"

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_image_input_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the background removal endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raw_img_data = requests.get(inputs["image_url"]).content

        tmp_local_image_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        tmp_local_image_file.write(raw_img_data)

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files={"image": open(tmp_local_image_file.name, "rb")},
        )

        tmp_local_image_file.close()
        os.unlink(tmp_local_image_file.name)

        assert response.status_code == 200

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    @pytest.mark.parametrize("inputs", inputs_to_test)
    def test_image_url_input_task(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the background removal endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={"image_url": inputs["image_url"]},
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an invalid image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        tmp_local_video_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        urlretrieve(
            f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4", tmp_local_video_file.name
        )

        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                files={
                    "image": (
                        "test.mp3",
                        open(tmp_local_video_file.name, "rb"),
                        "audio/mpeg",
                    )
                },
            )

            # assert response.status_code != 200 # TODO

        tmp_local_video_file.close()
        os.unlink(tmp_local_video_file.name)

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_url_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an invalid image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        with pytest.raises(Exception):
            response = client.post(
                url=self.target_url,
                params={"model": model} if model else {},
                data={"image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4"},
            )

            # assert response.status_code != 200 # TODO

    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an empty input

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

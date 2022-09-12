import os

import pytest
from fastapi.testclient import TestClient

from main import app
from tests.constants import HOST_TO_EXAMPLE_STORAGE, PATH_TO_EXAMPLE_FILES
from tests.utils import get_models_to_test

client = TestClient(app)
models = get_models_to_test(os.path.split(__file__)[0])


class TestBackgroundRemoval:
  """
  Class to test the background removal endpoint
  """
    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    def test_jpg_image_input_task(self, model):
        response = client.post(
            url=f"/image/image/background-removal/",
            params={"model": model} if model else {},
            files={
                "image": open(os.path.join(PATH_TO_EXAMPLE_FILES, "test.jpg"), "rb")
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("model", models)
    def test_png_image_input_task(self, model):
        response = client.post(
            url=f"/image/image/background-removal/",
            params={"model": model} if model else {},
            files={
                "image": open(os.path.join(PATH_TO_EXAMPLE_FILES, "test.png"), "rb")
            },
        )

        assert response.status_code == 200

    @pytest.mark.mandatory
    @pytest.mark.parametrize("model", models)
    def test_jpg_image_url_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """
        response = client.post(
            url=f"/image/image/background-removal/",
            params={"model": model} if model else {},
            data={
                "image_url": f"{HOST_TO_EXAMPLE_STORAGE}/examples/image/image/background-removal/owl2.jpg"
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("model", models)
    def test_png_image_url_input_task(self, model):
        response = client.post(
            url=f"/image/image/background-removal/",
            params={"model": model} if model else {},
            data={
                "image_url": f"{HOST_TO_EXAMPLE_STORAGE}/examples/image/image/background-removal/owl2.png"
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("model", models)
    def test_invalid_image_input_task(self, model):
        with pytest.raises(Exception):
            response = client.post(
                url=f"/image/image/background-removal/",
                params={"model": model} if model else {},
                files={
                    "image": (
                        "test.mp3",
                        open(os.path.join(PATH_TO_EXAMPLE_FILES, "test.mp3"), "rb"),
                        "audio/mpeg",
                    )
                },
            )

            # assert response.status_code != 200 # TODO

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
                url=f"/image/image/background-removal/",
                params={"model": model} if model else {},
                data={"image_url": f"{HOST_TO_EXAMPLE_STORAGE}/test/test.mp4"},
            )

            # assert response.status_code != 200 # TODO

    @pytest.mark.parametrize("model", models)
    def test_empty_input_task(self, model):
        response = client.post(
            url=f"/image/image/background-removal/",
            params={"model": model} if model else {},
            data={},
        )

        assert response.status_code == 200  # TODO: change to != 200

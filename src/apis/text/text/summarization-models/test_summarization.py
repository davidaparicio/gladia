import pytest
import os

import requests

from main import app
from tests import create_default_tests
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(
    ["text", "source_language", "min_length", "max_length"]
)


class TestsSummarization(
    create_default_tests(
        class_name="BasicTestsSummarization",
        client=requests,
        target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '0.0.0.0')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/text/text/summarization/",
        models_to_test=models,
        inputs_to_test=inputs_to_test,
    )
):
    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_invalid_text_param(self, model):
        """
        Test the headline generation endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "text": bytes(4),
                "source_language": inputs_to_test[0]["source_language"],
                "min_length": inputs_to_test[0]["min_length"],
                "max_length": inputs_to_test[0]["max_length"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_invalid_source_language_param(self, model):
        """
        Test the headline generation endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "text": inputs_to_test[0]["text"],
                "source_language": bytes(4),
                "min_length": inputs_to_test[0]["min_length"],
                "max_length": inputs_to_test[0]["max_length"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_min_length_param(self, model):
        """
        Test the headline generation endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "text": inputs_to_test[0]["text"],
                "source_language": inputs_to_test[0]["source_language"],
                "min_length": "some random value",
                "max_length": inputs_to_test[0]["max_length"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_max_length_param(self, model):
        """
        Test the headline generation endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={
                "text": inputs_to_test[0]["text"],
                "source_language": inputs_to_test[0]["source_language"],
                "min_length": inputs_to_test[0]["min_length"],
                "max_length": "some random value",
            },
        )

        assert response.status_code == 422

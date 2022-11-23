import os

import pytest
import requests

from tester import create_default_tests
from tester.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["text", "max_length"])


class TestsHeadlineGeneration(
    create_default_tests(
        class_name="BasicTestsHeadlineGeneration",
        client=requests,
        target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8080'))}/text/text/headline-generation/",
        models_to_test=models,
        inputs_to_test=inputs_to_test,
    )
):
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
                "max_length": "some random value",
            },
        )

        assert response.status_code == 422

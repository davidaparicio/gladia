import os

import pytest
import requests

from main import app
from tests import create_default_tests
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(
    ["input_string", "source_language", "target_language"]
)


class TestsTranslation(
    create_default_tests(
        class_name="BasicTestsTranslation",
        client=requests,
        target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/text/text/translation/",
        models_to_test=models,
        inputs_to_test=inputs_to_test,
    )
):
    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_invalid_input_string_param(self, model):
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
                "input_string": bytes(4),
                "source_language": inputs_to_test[0]["source_language"],
                "target_language": inputs_to_test[0]["target_language"],
            },
        )

        assert response.status_code == 422

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
                "input_string": inputs_to_test[0]["input_string"],
                "source_language": bytes(4),
                "target_language": inputs_to_test[0]["target_language"],
            },
        )

        assert response.status_code in [422, 500]

    @pytest.mark.parametrize("model", models)
    def test_invalid_target_language_param(self, model):
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
                "input_string": inputs_to_test[0]["input_string"],
                "source_language": inputs_to_test[0]["source_language"],
                "target_language": bytes(4),
            },
        )

        assert response.status_code in [422, 500]

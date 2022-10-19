import os
import pytest
import requests

from tests import create_default_tests
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(
    ["input_string_language_1", "input_string_language_2"]
)


class TestsWordAlignment(
    create_default_tests(
        class_name="BasicTestsWordAlignment",
        client=requests,
        target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '0.0.0.0')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/text/text/word-alignment/",
        models_to_test=models,
        inputs_to_test=inputs_to_test,
    )
):
    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")
    @pytest.mark.parametrize("model", models)
    def test_invalid_input_string_language_1_param(self, model):
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
                "input_string_language_1": bytes(4),
                "input_string_language_2": inputs_to_test[0]["input_string_language_2"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")
    @pytest.mark.parametrize("model", models)
    def test_invalid_input_string_language_2_param(self, model):
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
                "input_string_language_1": inputs_to_test[0]["input_string_language_1"],
                "input_string_language_2": bytes(4),
            },
        )

        assert response.status_code == 422

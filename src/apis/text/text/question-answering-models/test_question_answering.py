import os

import pytest
import requests

from main import app
from tests import create_default_tests
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["context", "question", "top_k"])


class TestsQuestionAnswering(
    create_default_tests(
        class_name="BasicTestsQuestionAnswering",
        client=requests,
        target_url=f"http://{os.getenv('TEST_CLIENT_HOST', '127.0.0.1')}:{int(os.getenv('TEST_CLIENT_PORT', '8000'))}/text/text/question-answering/",
        models_to_test=models,
        inputs_to_test=inputs_to_test,
    )
):
    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_invalid_context_param(self, model):
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
                "context": bytes(4),
                "question": inputs_to_test[0]["question"],
                "top_k": inputs_to_test[0]["top_k"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.skip("Model neither crash nor returns a 422/500 status code")  # FIXME
    @pytest.mark.parametrize("model", models)
    def test_invalid_question_param(self, model):
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
                "context": inputs_to_test[0]["context"],
                "question": bytes(4),
                "top_k": inputs_to_test[0]["top_k"],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("model", models)
    def test_invalid_top_k_param(self, model):
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
                "context": inputs_to_test[0]["context"],
                "question": inputs_to_test[0]["question"],
                "top_k": "some random value",
            },
        )

        assert response.status_code == 422

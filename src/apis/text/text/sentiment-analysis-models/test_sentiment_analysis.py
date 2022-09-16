from fastapi.testclient import TestClient

from main import app
from tests import create_default_text_to_text_tests
from tests.utils import get_inputs_to_test, get_models_to_test

TestSentimentAnalysis = create_default_text_to_text_tests(
    class_name="TestSentimentAnalysis",
    client=TestClient(app),
    target_url="/text/text/sentiment-analysis/",
    models_to_test=get_models_to_test(),
    inputs_to_test=get_inputs_to_test(["text"]),
)

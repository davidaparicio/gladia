from fastapi.testclient import TestClient

from main import app
from tests import create_default_text_to_text_tests
from tests.utils import get_inputs_to_test, get_models_to_test

models = get_models_to_test()
inputs_to_test = get_inputs_to_test(["text"])


TestsConversationSummarization = create_default_text_to_text_tests(
    class_name="TestsConversationSummarization",
    client=TestClient(app),
    target_url="/text/text/conversation-summarization/",
    models_to_test=models,
    inputs_to_test=inputs_to_test,
)

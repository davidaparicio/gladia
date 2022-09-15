from fastapi.testclient import TestClient

from main import app
from tests import create_default_image_to_image_tests
from tests.utils import get_inputs_to_test, get_models_to_test


TestUncolorization = create_default_image_to_image_tests(
    class_name="TestUncolorization",
    client=TestClient(app),
    target_url="/image/image/uncolorization/",
    models_to_test=get_models_to_test(),
    inputs_to_test=get_inputs_to_test(["image_url"])
)

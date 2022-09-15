from fastapi.testclient import TestClient

from main import app
from tests import create_default_image_to_image_tests
from tests.utils import get_inputs_to_test, get_models_to_test

TestSuperResolution = create_default_image_to_image_tests(
    class_name="TestSuperResolution",
    client=TestClient(app),
    target_url="/image/image/super-resolution/",
    models_to_test=get_models_to_test(),
    inputs_to_test=get_inputs_to_test(["image_url"]),
)

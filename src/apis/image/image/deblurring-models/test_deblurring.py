import os
from fastapi.testclient import TestClient

from main import app
from tests import create_default_image_to_image_tests
from tests.utils import get_inputs_to_test, get_models_to_test

TestDeblurring = create_default_image_to_image_tests(
    class_name="TestDeblurring",
    client=TestClient(app),
    target_url="/image/image/deblurring/",
    models_to_test=get_models_to_test(os.path.split(__file__)[0]),
    inputs_to_test=get_inputs_to_test(os.path.split(__file__)[0], ["image_url"])
)

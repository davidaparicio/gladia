import tempfile
import warnings
from copy import deepcopy
from logging import getLogger
from typing import Any, Callable, Dict, List

import nvidia_smi
import psutil
import pytest
import requests

logger = getLogger(__name__)


@pytest.fixture(autouse=True)
def memory_usage(TestSuite, record_testsuite_property):

    # Code run before the test
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    ram_before_test = psutil.virtual_memory().percent
    vram_before_test = info.free

    yield

    # Code run after the test
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    ram_after_test = psutil.virtual_memory().percent
    vram_after_test = info.free

    record_testsuite_property(
        f"{TestSuite} delta RAM", ram_after_test - ram_before_test
    )
    record_testsuite_property(
        f"{TestSuite} delta VRAM", vram_after_test - vram_before_test
    )

    logger.info(f"{ram_before_test=}")
    logger.info(f"{ram_after_test=}")

    logger.info(f"{vram_before_test=}")
    logger.info(f"{vram_after_test=}")

    if abs(ram_after_test - ram_before_test) >= 100_000:
        logger.warning(
            f"{TestSuite} seems to have a memory leak {abs(ram_after_test - vram_before_test)=}"
        )
        warnings.warn(
            f"{TestSuite} seems to have a memory leak {abs(ram_after_test - vram_before_test)=}"
        )

    if abs(vram_after_test - vram_before_test) >= 100_000:
        logger.warning(
            f"{TestSuite} seems to have a memory leak {abs(vram_after_test - vram_before_test)=}"
        )
        warnings.warn(
            f"{TestSuite} seems to have a memory leak {abs(vram_after_test - vram_before_test)=}"
        )

    # TODO: after switch to assert
    # assert abs(ram_after_test - ram_before_test) < 250_000, f"{TestSuite} seems to have a memory leak {abs(ram_after_test - vram_before_test)=}"
    # assert abs(vram_after_test - vram_before_test) < 250_000, f"{TestSuite} seems to have a memory leak {abs(vram_after_test - vram_before_test)=}"

    nvidia_smi.nvmlShutdown()


def __apply_decorators(func, *decorators):
    def deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return deco(func)


def get_test_correct_inputs(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

    def __test_correct_inputs(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        files = {}
        data = {}

        for input_name in inputs:
            if input_name.endswith("_url"):
                tmp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
                tmp_file.write(requests.get(inputs[input_name]).content)

                files[input_name.replace("_url", "")] = open(tmp_file.name, "rb")
            else:
                data[input_name] = inputs[input_name]

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            files=files,
            data=data,
        )

        assert (
            response.status_code == 200
        ), f"client returned a non 200 status code: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_correct_inputs,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_correct_inputs_url(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

    def __test_correct_inputs_url(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data=inputs,
        )

        assert (
            response.status_code == 200
        ), f"client returned a non 200 status code: {response.status_code} with the following message: {response.content}"

    return __apply_decorators(
        __test_correct_inputs_url,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_invalid_inputs_url(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

    def __test_invalid_inputs_url(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        for input_name in inputs:
            if input_name.endswith("_url"):
                inputs[input_name] = "http://some/random/url/that/does/not.exists"

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data=inputs,
        )

        assert response.status_code in [
            400,
            422,
        ], f"(target: {self.target_url})exepceted status code to be 400/422 but received: {response.status_code}."

    return __apply_decorators(
        __test_invalid_inputs_url,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_invalid_params(
    models_to_test: List[str], inputs_to_test: List[Dict[str, Any]]
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Generate the test function for basic image inputs

    Args:
        models_to_test (List[str]): models to test
        inputs_to_test (List[Dict[str, Any]]): inputs to test the model with

    Returns:
        Callable[[str, Dict[str, Any]], bool]: test function
    """

    def __test_invalid_params(self, model: str, inputs: Dict[str, Any]) -> bool:
        """
        Test the endpoint with a jpg image input

        Args:
            model (str): model to test
            inputs (Dict[str, Any]): input values to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        is_able_to_test_invalid_param = False

        for input_name in inputs:
            if input_name.endswith("_url"):
                continue

            if type(inputs[input_name]) in (int, float):
                inputs[input_name] = "some random value"
                is_able_to_test_invalid_param = True

            # This will not raise an error
            # TODO: find a way to provid invalid value when inputs[input_name]) is str
            elif type(inputs[input_name]) is str:
                inputs[input_name] = 123

        if not is_able_to_test_invalid_param:
            pytest.skip(
                "Unable to test the model with invalid parameters. Skipping this test..."
            )

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data=inputs,
        )

        assert response.status_code in [
            400,
            422,
        ], f"exepceted status code to be 400/422 but received: {response.status_code}."

    return __apply_decorators(
        __test_invalid_params,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
        pytest.mark.parametrize("inputs", inputs_to_test),
    )


def get_test_empty_input_task(models_to_test: List[str]) -> Callable[[str], bool]:
    """
    Generate the test function testing the endpoint with an empty input

    Args:
        models_to_test (List[str]): models to test

    Returns:
        Callable[[str], bool]: test function
    """

    def __test_empty_input_task(self, model: str) -> bool:
        """
        Test the endpoint with an empty input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        response = self.client.post(
            url=self.target_url,
            params={"model": model} if model else {},
            data={},
        )

        assert response.status_code in [
            400,
            422,
        ], f"exepceted status code to be 400/422 but received: {response.status_code}. Body: {response.content}"

    return __apply_decorators(
        __test_empty_input_task,
        pytest.mark.mandatory(),
        pytest.mark.parametrize("model", models_to_test),
    )


def create_default_tests(
    class_name: str,
    client,
    target_url: str,
    models_to_test: List[str],
    inputs_to_test: List[Dict[str, Any]],
):

    file_in_inputs = any(
        [input_name.endswith("_url") for input_name in inputs_to_test[0]]
    )
    only_files_as_input = all(
        [input_name.endswith("_url") for input_name in inputs_to_test[0]]
    )

    DefaultTestSuite = type(
        class_name,
        (),
        {
            "client": client,
            "target_url": target_url,
            "test_correct_inputs": get_test_correct_inputs(
                deepcopy(models_to_test), deepcopy(inputs_to_test)
            ),
            "memory_usage": memory_usage,
            "test_empty_input_task": get_test_empty_input_task(
                deepcopy(models_to_test),
            ),
        },
    )

    if file_in_inputs:
        setattr(
            DefaultTestSuite,
            "test_correct_inputs_url",
            get_test_correct_inputs_url(
                deepcopy(models_to_test), deepcopy(inputs_to_test)
            ),
        )

        setattr(
            DefaultTestSuite,
            "test_invalid_inputs_url",
            get_test_invalid_inputs_url(
                deepcopy(models_to_test), deepcopy(inputs_to_test)
            ),
        )

    if not only_files_as_input:
        setattr(
            DefaultTestSuite,
            "test_invalid_params",
            get_test_invalid_params(deepcopy(models_to_test), deepcopy(inputs_to_test)),
        )

    return DefaultTestSuite

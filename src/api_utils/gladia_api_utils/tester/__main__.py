import pytest

from .test_autogenerated import __create_autogenerated_tests

if __name__ == "__main__":

    retcode = pytest.main(
        [
            __file__,
            "-o",
            "log_cli=true",
            "--log-cli-level=DEBUG",
        ]
    )

    assert retcode == 0, f"return code should be 0 but received {retcode}"

else:
    TestAutogenerated = __create_autogenerated_tests()

from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """Define custome command line arguments

    Args:
        parser (Parser): parser used by pytest to retrieve command line arguments
    """

    parser.addoption("--default-models-only", action="store_true", default=False)

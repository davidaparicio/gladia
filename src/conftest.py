from _pytest.config.argparsing import Parser

gladia_server_process = None


def pytest_addoption(parser: Parser) -> None:
    """
    Define custom command line arguments

    Args:
        parser (Parser): parser used by pytest to retrieve command line arguments

    Returns:
        None
    """

    parser.addoption("--default-models-only", action="store_true", default=False)
    parser.addoption("--default-inputs-only", action="store_true", default=False)
    parser.addoption("--deactivate-automated-tests", action="store_true", default=False)


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    import os
    from multiprocessing import Process

    import uvicorn

    from main import app

    global gladia_server_process

    gladia_server_process = Process(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": os.getenv("TEST_CLIENT_HOST", "0.0.0.0"),
            "port": int(os.getenv("TEST_CLIENT_PORT", "8000")),
        },
    )

    gladia_server_process.start()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """

    global gladia_server_process

    if gladia_server_process is not None:
        gladia_server_process.kill()

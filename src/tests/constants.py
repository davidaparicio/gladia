import os

HOST_TO_EXAMPLE_STORAGE = os.getenv("HOST_TO_EXAMPLE_STORAGE", "http://files.gladia.io")
PATH_TO_EXAMPLE_FILES = os.getenv(
    "PATH_TO_EXAMPLE_FILES", os.path.join(os.getenv("PATH_TO_GLADIA_SRC"), "unit-test")
)

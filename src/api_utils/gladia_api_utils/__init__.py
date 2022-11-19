import json
import logging
import os
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from .add_routes_to_router import add_routes_to_router
from .get_activated_task_path import get_activated_task_path
from .secret_management import SECRETS

log_path = os.getenv("API_UTILS_LOGGING_PATH", "./.api_utils.logs")

logging_format = os.getenv(
    "API_UTILS_LOGGING_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging_level = {
    None: logging.NOTSET,
    "": logging.NOTSET,
    "none": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}.get(os.getenv("API_UTILS_LOGGING_LEVEL", "info"), logging.INFO)

logging.basicConfig(
    level=logging_level,
    format=logging_format,
)

logger = logging.getLogger(__name__)


if "PATH_TO_GLADIA_SRC" in os.environ and "API_UTILS_LOGGING_PATH" not in os.environ:
    # check if the config.json file is present in the gladia src folder
    config_path = os.path.join(os.environ["PATH_TO_GLADIA_SRC"], "config.json")
    if os.path.exists(config_path):
        # read the config.json file and get the logging level and format
        with open(config_path) as f:
            config = json.load(f)
            logging_timing_activated = config.get("logs.timing_activated", True)
            logging_level = config.get("logs.log_level", logging_level)
            log_path = config.get("logs.log_path", log_path)
            logging_format = config.get("logs.log_format", logging_format)


rotating_file_handler = RotatingFileHandler(
    log_path,
    maxBytes=4096,
    backupCount=1,
)

rotating_file_handler.setFormatter(logging.Formatter(logging_format))
rotating_file_handler.setLevel(logging_level)

stream_handler = StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging_format))
stream_handler.setLevel(logging_level)

logger.addHandler(rotating_file_handler)
logger.addHandler(stream_handler)

__all__ = ["get_activated_task_path", "SECRETS"]

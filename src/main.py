import json
import logging
import os
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi_utils.timing import add_timing_middleware
from gladia_api_utils import add_routes_to_router
from gladia_api_utils.apis_for_subprocess import build_all_subprocesses_apis
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import RedirectResponse

import apis


def __init_config() -> dict:
    """
    Load config file and return it as a dict.
    Default path is `config.json`, use API_CONFIG_FILE varenv to change it.

    Args:
        None

    Returns:
        dict: config dict
    """

    config_file = os.getenv("API_CONFIG_FILE", "config.json")

    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            return json.load(f)


def __init_logging(api_config: dict) -> logging.Logger:
    """
    Create a logging.Logger with it format set to config["logs"]["log_format"] f exist, else default.

    Args:
        api_config (dict): config dict

    Returns:
        logging.Logger: logger
    """

    logging_level = {
        None: logging.NOTSET,
        "": logging.NOTSET,
        "none": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(api_config["logs"]["log_level"], logging.INFO)

    logging.basicConfig(
        level=logging_level,
        format=api_config["logs"]["log_format"],
    )

    logger = logging.getLogger(__name__)

    rotating_file_handler = RotatingFileHandler(
        api_config["logs"]["log_path"],
        maxBytes=100_000,
        backupCount=10,
    )
    rotating_file_handler.setFormatter(
        logging.Formatter(api_config["logs"]["log_format"])
    )
    rotating_file_handler.setLevel(logging_level)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(logging.Formatter(api_config["logs"]["log_format"]))
    stream_handler.setLevel(logging_level)

    logger.addHandler(rotating_file_handler)
    logger.addHandler(stream_handler)

    existing_loggers = [logging.getLogger()]
    existing_loggers += [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    set_logger_handlers = set(logger.handlers)

    for external_logger in existing_loggers:
        if "gladia_api_utils" in external_logger.name:
            continue

        for handler in external_logger.handlers:
            handler.setLevel(logging_level)

        # Prevents having multiple times the same handler
        external_logger.handlers = list(
            set(external_logger.handlers).union(set(set_logger_handlers))
        )

    return logger


def __init_prometheus_instrumentator(instrumentator_config: dict) -> Instrumentator:
    """
    Initialize the prometheus_fastapi_instrumentator.Instrumentator using api config dict

    Args:
        instrumentator_config (dict): config dict

    Returns:
        Instrumentator: Initialized Instrumentator
    """

    return Instrumentator(
        should_group_status_codes=instrumentator_config["should_group_status_codes"],
        should_ignore_untemplated=instrumentator_config["should_ignore_untemplated"],
        should_group_untemplated=instrumentator_config["should_group_untemplated"],
        should_respect_env_var=instrumentator_config["should_respect_env_var"],
        env_var_name=instrumentator_config["env_var_name"],
        excluded_handlers=instrumentator_config["excluded_handlers"],
        should_round_latency_decimals=instrumentator_config[
            "should_round_latency_decimals"
        ],
        round_latency_decimals=instrumentator_config["round_latency_decimals"],
        should_instrument_requests_inprogress=instrumentator_config[
            "should_instrument_requests_inprogress"
        ],
        inprogress_name=instrumentator_config["inprogress_name"],
        inprogress_labels=instrumentator_config["inprogress_labels"],
    )


def __set_app_middlewares(api_app: FastAPI, api_config: dict) -> None:
    """
    Set up the api middlewares

    Args:
        api_app (FastAPI): FastAPI representing the API
        api_config (dict): config dict telling which middlewares to use

    Returns:
        None
    """

    if api_config["logs"]["timing_activated"]:
        add_timing_middleware(api_app, record=logging.info, prefix="app")

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config["CORS"]["allow_origins"],
        allow_credentials=api_config["CORS"]["allow_credentials"],
        allow_methods=api_config["CORS"]["allow_methods"],
        allow_headers=api_config["CORS"]["allow_headers"],
    )


nltk.download("punkt")

config = __init_config()
logger = __init_logging(config)

app = FastAPI(default_response_class=ORJSONResponse)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health", include_in_schema=False)
async def health():
    return None


__set_app_middlewares(app, config)

if config["prometheus"]["active"]:
    instrumentator = __init_prometheus_instrumentator(
        config["prometheus"]["instrumentator"]
    )


build_all_subprocesses_apis()

add_routes_to_router(
    app=app, apis_folder_name="apis", active_tasks=config["active_tasks"], package=apis
)

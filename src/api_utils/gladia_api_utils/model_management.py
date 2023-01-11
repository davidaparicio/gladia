import os
import shutil
import sys
import threading
from logging import getLogger
from pathlib import Path
from urllib.parse import urlparse

import spacy
from git import Repo
from spacy.cli.download import get_compatibility

from .file_management import (
    delete_directory,
    delete_file,
    download_file,
    get_tmp_filename,
    is_uncompressable,
    uncompress,
)
from .system import load_models_config

logger = getLogger(__name__)


GLADIA_MODEL_PATH = os.getenv("GLADIA_MODEL_PATH", "/gladia/models")
GLADIA_SRC_PATH = os.getenv("GLADIA_SRC_PATH", "/app")
SPACY_CACHE_DIR = os.getenv("SPACY_CACHE_DIR", "/gladia/spacy/models")

models_config = load_models_config()

SPACY_LANGUAGE_MODEL = models_config["spacy"]["models"]


def get_spacy_cache_dir_for_model(model: str) -> str:
    """Get the spaCy cache directory for the specified model.

    Args:
        model (str): The name of the spaCy model to get the cache directory for.

    Returns:
        str: The spaCy cache directory for the specified model.
    """
    return os.path.join(SPACY_CACHE_DIR, model)


def get_spacy_language_code(language_code: str) -> str:
    """Get the spaCy language code for the specified language.

    Args:
        language_code (str): The 639-3 language code of the language to get the spaCy language code for. (e.g. "eng")

    Returns:
        str: The spaCy language code for the specified language.
    """
    spacy_models = get_compatibility()
    if language_code in SPACY_LANGUAGE_MODEL:
        language_code = SPACY_LANGUAGE_MODEL[language_code]["model"][:2]
    elif language_code not in spacy_models:
        language_code = "xx"

    return language_code


def load_spacy_language_model(language: str) -> spacy.language.Language:
    """Load the spaCy natural language processing model for the specified language.

    Args:
        language_code: The 639-3 language code of the language to load the model for.

    Returns:
        The spaCy language model for the specified language.
    """
    language_model = None
    spacy_models = get_compatibility()
    if language in spacy_models:
        language_model = language
    elif language not in SPACY_LANGUAGE_MODEL:
        language_model = SPACY_LANGUAGE_MODEL["others"]["model"]
    else:
        language_model = SPACY_LANGUAGE_MODEL[language]["model"]

    try:
        nlp = spacy.load(os.path.join(SPACY_CACHE_DIR, language_model))
    except ModuleNotFoundError:
        logger.info(f"Download spacy model {language_model}")
        spacy.cli.download(language_model)
        nlp = spacy.load(language_model)
        nlp.to_disk(os.path.join(SPACY_CACHE_DIR, language_model))

    return nlp


def __download_huggingface_model(
    url: str,
    output_path: str,
    reset: bool = False,
    uncompress_after_download: bool = True,
) -> bool:
    """
    Download a model from huggingface and uncompress it if necessary.
    Return True if the model was an huggingface model, False otherwise

    Args:
        url (str): url of the model to download
        output_path (str): path to download the model to

    Returns:
        bool: True if the model was an huggingface model, False otherwise
    """

    domain = urlparse(url).netloc

    is_hugging_face = False
    # if domain is huggingface
    # and if its not a file (resolve) but a git-lfs repo
    # else if (resolve) or not huggingface consider url as a file
    if "huggingface.co" in domain:
        is_hugging_face = True
        if "/resolve/" not in url:
            # check if directory exists if not clone it else pull
            os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"

            if not os.path.isdir(Path(output_path)):
                logger.debug(f"Cloning HuggingFace Model from {url}")
                Repo.clone_from(url, output_path)
                os.system(f"cd {output_path} && git lfs pull")

            else:
                if reset:
                    logger.debug(f"Pulling HuggingFace Model from {url}")
                    repo = Repo(output_path)
                    repo.git.reset("--hard", "origin/main")
                    os.system(f"cd {output_path} && git lfs pull")
        else:
            __download_and_uncompress_model(
                url, output_path, uncompress_after_download=uncompress_after_download
            )

    return is_hugging_face


def __download_and_uncompress_model(
    url: str, output_path: str, uncompress_after_download: bool = True
) -> None:
    logger.info(f"Downloading {url}")

    # if the output_path is not an existing directory create it
    if not os.path.exists(Path(output_path).parent):
        os.makedirs(Path(output_path).parent, exist_ok=True)

    # create a temporary folder to download the model to
    dl_tmp_filepath = get_tmp_filename()
    uncompress_tmp_dirpath = get_tmp_filename()

    logger.debug(f"Temporary filepath for download: {dl_tmp_filepath}")

    downloaded_full_path = download_file(
        url=url,
        file_full_path=dl_tmp_filepath,
        force_create_dir=True,
        force_redownload=False,
    )
    logger.debug(f"Downloaded model to {downloaded_full_path}")

    # if the model is uncompressable uncompress it
    if uncompress_after_download and is_uncompressable(str(downloaded_full_path)):
        logger.info("Uncompressing {downloaded_full_path} to {output_path}")

        uncompress(
            path=downloaded_full_path,
            destination=uncompress_tmp_dirpath,
            delete_after_uncompress=True,
        )

        logger.info(
            f"Uncompressed model from {uncompress_tmp_dirpath} to {output_path}"
        )

        # move the uncompress model to the output path
        shutil.move(uncompress_tmp_dirpath, output_path)
    else:
        logger.info(f"Moving {dl_tmp_filepath} to {output_path}")
        shutil.move(dl_tmp_filepath, output_path)

    # clean up temporary folder
    delete_file(dl_tmp_filepath)
    delete_directory(uncompress_tmp_dirpath)


def create_folder_in_model_cache_directory(folder_path: str) -> str:
    """
    Create a folder withing a model cache directory
    No absolute path should be provided

    Args:
        folder_path (str): absolute path to the folder to create in the model cache directory

    Returns:
        str: absolute path to the folder created
    """
    if not os.path.isabs(folder_path):
        namespace = sys._getframe(1).f_globals
        rel_path = str(os.path.dirname(namespace["__file__"]))

        folder_path = GLADIA_MODEL_PATH + rel_path + "/" + folder_path
        logger.debug(f"Relative path detected, using {folder_path} as path")

    else:
        raise ValueError("Absolute path provided")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    return folder_path


def download_model(
    url: str,
    output_path: str,
    uncompress_after_download: bool = True,
    file_type: str = None,
    reset: bool = True,
    force_redownload: bool = False,
    branch: str = "origin",
) -> str:
    """
    Download a model and uncompress it if necessary
    reset lets you decide not to force sync between huggingface hub and you local repo (for testing purposes for instance)

    Args:
        url (str): url of the model to download
        output_path (str): path to download the model to
        uncompress_after_download (bool): whether to uncompress the model after download (default: True)
        file_type (str): type of file to download (if None, will try to guess)
        reset (bool): whether to reset the local repo (default: True)
        branch (str): branch to use (default: origin)

    Returns:
        str: path to the downloaded model
    """

    if not os.path.isabs(output_path):
        namespace = sys._getframe(1).f_globals
        rel_path = str(os.path.dirname(namespace["__file__"]))

        output_path = GLADIA_MODEL_PATH + rel_path + "/" + output_path
        logger.debug(f"Relative path detected, using {output_path} as output path")

    if not os.path.exists(Path(output_path).parent):
        os.makedirs(Path(output_path).parent, exist_ok=True)

    logger.debug(f"Downloading model from {url} to {output_path}")

    if (
        not os.path.exists(output_path)
        or force_redownload
        and not __download_huggingface_model(
            url=url,
            output_path=output_path,
            reset=reset,
            uncompress_after_download=uncompress_after_download,
        )
    ):
        __download_and_uncompress_model(
            url=url,
            output_path=output_path,
            uncompress_after_download=uncompress_after_download,
        )

    return output_path


def download_models(model_list: dict) -> dict:
    """
    Download a list of models and uncompress them if necessary

    Args:
        model_list (dict): list of models to download should be [(url, output_path, uncompression_mode)]

    Returns:
        dict: list of models with their paths
    """

    # manage relative imports
    namespace = sys._getframe(1).f_globals

    rel_path = namespace["__file__"]
    rel_path = rel_path.lstrip("./")
    if ".py" in rel_path:
        rel_path = os.path.dirname(rel_path)

    logger.debug("Downloading multiple models")
    threads = []
    output = dict()

    for key, model in model_list.items():
        if not os.path.isabs(model["output_path"]):
            model["output_path"] = os.path.join(
                GLADIA_MODEL_PATH, rel_path, model["output_path"]
            )

            t = threading.Thread(
                target=download_model, args=(model["url"], model["output_path"])
            )
            output[key] = model
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    return output

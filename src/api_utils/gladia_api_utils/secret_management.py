import os

from warnings import warn
from logging import getLogger

logger = getLogger(__name__)


class SecretManager:

    __secrets = {
        "HUGGINGFACE_ACCESS_TOKEN": {
            "value": os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
            "error_message": "HUGGINGFACE_ACCESS_TOKEN environement variable was not set, please visite hugging-face's web site to retrieve your access token : (https://huggingface.co/settings/tokens) (https://huggingface.co/docs/hub/security-tokens/)"
        }
    }

    def __getitem__(self, item):
        if item not in self.__secrets:
            error_message = f"{item} is not set in the SecretManager."

            logger.error(error_message)
            raise RuntimeError(error_message)
        
        if self.__secrets[item]["value"] is None:
            warn(self.__secrets[item]["error_message"])
            logger.warning(self.__secrets[item]["error_message"])

            return None

        return self.__secrets[item]["value"]


SECRETS = SecretManager()

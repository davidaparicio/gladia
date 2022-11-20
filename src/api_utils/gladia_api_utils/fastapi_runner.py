import os
import sys
from asyncio.log import logger
from importlib import import_module, util
from typing import Any

import uvicorn
from fastapi import FastAPI, Request

os.system("pip install requests_toolbelt")
from pathlib import Path

from gladia_api_utils.submodules import clean_kwargs_based_on_router_inputs
from requests_toolbelt.multipart import decoder

app = FastAPI()

# print current working directory


@app.get("/status")
def status() -> int:
    """
    Dummy endpoint to check if the API is running

    Args:
        None

    Returns:
        int: 200
    """
    return 200


@app.post("/predict")
async def apply(request: Request) -> Any:
    """
    Passthrough to the model's predict function

    Args:
        payload (dict): The payload to pass to the model's predict function

    Returns:
        Any: The result of the model's predict function
    """
    multipart_string = await request.body()

    payload = {}

    if "multipart/form-data" in request.headers["Content-Type"]:
        # Decode the multipart string to a dictionnary
        multipart_data = decoder.MultipartDecoder(
            multipart_string, request.headers["Content-Type"]
        )

        for part in multipart_data.parts:
            key = (
                part.headers[b"Content-Disposition"]
                .decode()
                .split("=")[1][1:-1]
                .split('"')[0]
            )
            payload[key] = part.content

    global INPUTS
    kwargs, success, error_message = await clean_kwargs_based_on_router_inputs(
        payload, INPUTS
    )
    if not success:
        return {"error": error_message}

    output = model.predict(**kwargs)
    logger.error(output)

    return output


if __name__ == "__main__":
    """
    Dynamically import the model module and run the API on port defined by the
    argument passed to the script
    """

    # should be an int between 1 and 65535
    port = int(sys.argv[1])

    # model path should look like the fullpath to the model file
    # /app/src/apis/image/image/blabla/MyModel/MyModel.py
    module_path = sys.argv[2]

    # fix the dependency issue with the imported model
    # allow the model to import from it's own src folder
    sys.path.insert(0, os.getcwd())

    # dynamically import the model module
    spec = util.spec_from_file_location("model", module_path)
    model = util.module_from_spec(spec)
    sys.modules["model"] = model
    spec.loader.exec_module(model)

    # import the parent routeur module (task_routeur)
    # /app/src/apis/image/image/blabla.py
    # we will need this to get the inputs to clean the payload after
    # especially for the binary inputs to substitute the _url by the actual file
    # please refeer to the clean_kwargs_based_on_router_inputs function in gladia_api_utils.submodules
    # for more details
    models_folder = Path(module_path).parent.parent
    # extract the blabla part from models_folder
    root_package_path = models_folder.parent / models_folder.name.split("-")[0]

    # import the parent routeur module (task_routeur)
    # /app/src/apis/image/image/blabla.py
    # we will need this to get the inputs to clean the payload after
    # especially for the binary inputs to substitute the _url by the actual file
    # please refeer to the clean_kwargs_based_on_router_inputs function in gladia_api_utils.submodules
    # for more details
    PATH_TO_GLADIA_SRC = os.getenv("PATH_TO_GLADIA_SRC", "/app")
    sys.path.insert(0, PATH_TO_GLADIA_SRC)
    root_package_path = (
        str(root_package_path)
        .replace(PATH_TO_GLADIA_SRC, "")
        .replace("/", ".")
        .strip(".")
    )
    task_routeur = import_module(root_package_path)

    global INPUTS
    INPUTS = task_routeur.inputs

    del task_routeur
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

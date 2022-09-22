import os
os.system("pip install uvicorn fastapi")
from multiprocessing import Process
import uvicorn
from fastapi import FastAPI
import sys
from importlib import import_module

from multiprocessing import Process
import threading


clean_key = "tesseract-default".replace("-", "_")
model = import_module("tesseract-default")
# clean_key is used to avoid importlib.import_module to import the same module twice
# if the module is imported twice, the second import will fail
# this is a workaround to avoid this issue
# see https://stackoverflow.com/questions/8350853/how-to-import-module-when-module-name-has-a-dash-or-hyphen-in-it
if clean_key not in sys.modules:
    sys.modules[clean_key] = sys.modules["tesseract-default"]

app = FastAPI()

@app.get("/status")
def status():
    return 200

@app.post("/apply")
def apply(payload: dict={}):
    prediction = model.predict(**payload)
    return prediction

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=sys.argv[1])

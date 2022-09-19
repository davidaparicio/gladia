import os
import sys
import uvicorn
from fastapi import FastAPI
from importlib import import_module


app = FastAPI()

@app.get("/status")
def status():
    return 200

@app.post("/predict")
def apply(payload: dict={}):
    prediction = model.predict(**payload)
    return prediction

if __name__ == "__main__":
    port = int(sys.argv[1])
    module_path = sys.argv[2]
    model = import_module(module_path)
    uvicorn.run(app, host="127.0.0.1", port=port)

#!/bin/bash
MODE="${MODE:-standalone}"

API_SERVER_PORT_HTTP="${API_SERVER_PORT_HTTP:-8080}"
API_SERVER_WORKERS="${API_SERVER_WORKERS:-1}"
API_SERVER_TIMEOUT="${API_SERVER_TIMEOUT:-1200}"

rm /usr/bin/python3 && \
ln -s /usr/bin/python3.8 /usr/bin/python3

GLADIA_TMP_PATH="${GLADIA_TMP_PATH:-/tmp/gladia}"
GLADIA_TMP_MODEL_PATH="${GLADIA_TMP_MODEL_PATH:-$GLADIA_TMP_PATH/model}"

MII_CACHE_PATH="${MII_CACHE_PATH:-$GLADIA_TMP_MODEL_PATH/mii/cache}"
MII_MODEL_PATH="${MII_MODEL_PATH:-$GLADIA_TMP_MODEL_PATH/mii/models}"

GLADIA_TMP_TORCH_CACHE="${GLADIA_TMP_MODEL_PATH:-$GLADIA_TMP_PATH/torch_cache}"


mkdir -p $GLADIA_TMP_TORCH_CACHE
rm -rf /root/.cache/torch
ln -s $GLADIA_TMP_TORCH_CACHE /root/.cache/torch 

# init mii folders
mkdir -p $MII_CACHE_PATH
mkdir -p $MII_MODEL_PATH

service supervisor start

# initialize the nltk database
micromamba run -n server --cwd /app python prepare.py

micromamba run -n server --cwd /app gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP} \
--workers ${API_SERVER_WORKERS} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT}

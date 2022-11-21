#!/bin/bash
MODE="${MODE:-standalone}"

API_SERVER_PORT_HTTP="${API_SERVER_PORT_HTTP:-8080}"
API_SERVER_WORKERS="${API_SERVER_WORKERS:-1}"
API_SERVER_TIMEOUT="${API_SERVER_TIMEOUT:-1200}"

rm /usr/bin/python3 && \
ln -s /usr/bin/python3.8 /usr/bin/python3

service supervisor start

# initialize the nltk database
micromamba run -n server --cwd /app python -c "import nltk; nltk.download('punkt')"

micromamba run -n server --cwd /app gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP} \
--workers ${API_SERVER_WORKERS} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT}

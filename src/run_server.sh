#!/bin/bash
MODE="${MODE:-standalone}"

API_SERVER_PORT_HTTP="${API_SERVER_PORT_HTTP:-8080}"
API_SERVER_WORKERS="${API_SERVER_WORKERS:-1}"
API_SERVER_TIMEOUT="${API_SERVER_TIMEOUT:-1200}"

rm /usr/bin/python3 && \
ln -s /usr/bin/python3.8 /usr/bin/python3

# init nltk data used for truecase
service supervisor start

micromamba run -n server python -c "import nltk; nltk.download('punkt')"

if [ $MODE = "standalone" ]
then
  micromamba run -n server python warm_up.py
  micromamba run -n server tritonserver \
    --http-port ${TRITON_SERVER_PORT_HTTP} \
    --grpc-port ${TRITON_SERVER_PORT_GRPC} \
    --metrics-port ${TRITON_SERVER_PORT_METRICS} \
    --model-repository=${TRITON_MODELS_PATH} \
    --exit-on-error=false \
    --model-control-mode=explicit \
    --repository-poll-secs 10 \
    --allow-metrics=false & \
  gunicorn main:app \
    -b 0.0.0.0:${API_SERVER_PORT_HTTP} \
    --workers ${API_SERVER_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout ${API_SERVER_TIMEOUT}
    
elif [ $MODE = "server" ]
then
  micromamba run -n server gunicorn main:app \
  -b 0.0.0.0:${API_SERVER_PORT_HTTP} \
  --workers ${API_SERVER_WORKERS} \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout ${API_SERVER_TIMEOUT}
else
  echo "Error: "$MODE" in an unknown mode"
  exit 1
fi

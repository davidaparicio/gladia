#!/bin/bash
MODE="${MODE:-standalone}"

P="\e[36m"
G="\e[32m"
EC="\e[0m"

if [ -f $MAMBA_ROOT_PREFIX/envs/server/server.yml ]; then
        if ! cmp /app/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml; then
                micromamba update -f env.yaml
        else
                echo -e "${G}micromamba server up to date.${EC}"
        fi
else
        micromamba create -f env.yaml
        cp /app/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml
        # we need a better way to handle jax install, either in env or with dedicated check
        micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]==0.3.25\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
fi

export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

micromamba run -n server --cwd $VENV_BUILDER_PATH /bin/bash -c "python3 create_custom_envs.py --modality '.*' --debug_mode";

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/"

echo -e "${P}== FIXING PROTOBUH ==${EC}" 
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O $MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/google/protobuf/internal/builder.py

echo -e "${P}== ADJUSTING path rights ==${EC}" 
chown -R $DOCKER_USER:$DOCKER_GROUP $PATH_TO_GLADIA_SRC 
chown -R $DOCKER_USER:$DOCKER_GROUP $GLADIA_TMP_PATH 

echo -e "${P}== FIXING libcurl references ==${EC}" 
rm $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4 
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.6.0 $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4 
$CLEAN_LAYER_SCRIPT

echo -e "${P}== Starting supervisor ==${EC}"
service supervisor start

echo -e "${P}== Initialize the nltk database ==${EC}"
micromamba run -n server --cwd /app python warm_up.py

echo -e "${P}== Starting Gladia ==${EC}"
micromamba run -n server --cwd /app gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP:-8080} \
--workers ${API_SERVER_WORKERS:-1} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT:-1200}
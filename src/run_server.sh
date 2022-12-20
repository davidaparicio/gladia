#!/bin/bash
MODE="${MODE:-standalone}"

cat tools/version/lite.txt
echo build: $(cat tools/version/build)

P="\e[35m"
C="\e[36m"
G="\e[32m"
EC="\e[0m"
echo -e "${P}== INIT Micromamba Server Env ==${EC}"
if [ -f $MAMBA_ROOT_PREFIX/envs/server/server.yml ]; then
    if ! cmp /app/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml; then
        echo -e "${C}Updating micromamba server env.${EC}"
        micromamba update -f env.yaml
        # we need a better way to handle jax install, either in env or with dedicated check
        micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]==0.3.25\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        cp /app/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml
    else
        echo -e "${G}Micromamba server already up to date.${EC}"
    fi
else
    echo -e "${G}Creating micromamba server.${EC}"
    micromamba create -f env.yaml
    cp /app/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml
    # we need a better way to handle jax install, either in env or with dedicated check
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]==0.3.25\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
fi

export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

echo -e "${P}== INIT Micromamba Venvs ==${EC}"
micromamba run -n server --cwd $VENV_BUILDER_PATH /bin/bash -c "python3 create_custom_envs.py --modality '.*' --debug_mode";

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/"

echo -e "${P}== FIX Protobuh ==${EC}" 
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O $MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/google/protobuf/internal/builder.py

echo -e "${P}== ADJUST path rights ==${EC}" 
chown -R $DOCKER_USER:$DOCKER_GROUP $PATH_TO_GLADIA_SRC
chown -R $DOCKER_USER:$DOCKER_GROUP $GLADIA_TMP_PATH

echo -e "${P}== FIX libcurl references ==${EC}" 
rm $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.6.0 $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
$CLEAN_LAYER_SCRIPT

echo -e "${P}== START supervisor ==${EC}"
service supervisor start

echo -e "${P}== INIT nltk + Spacy ==${EC}"
micromamba run -n server --cwd /app python prepare.py

echo -e "${P}== START Gladia ==${EC}"
micromamba run -n server --cwd /app gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP:-8080} \
--workers ${API_SERVER_WORKERS:-1} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT:-1200}
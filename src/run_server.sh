#!/bin/bash
MODE="${MODE:-standalone}"

PATH_TO_GLADIA_SRC="${PATH_TO_GLADIA_SRC:-/app}"

GLADIA_PERSISTENT_PATH="${GLADIA_PERSISTENT_PATH:-/gladia}"

SPACY_CACHE_DIR="${SPACY_CACHE_DIR:-$GLADIA_PERSISTENT_PATH/spacy/models}"
SPACY_CACHE_PURGE="${SPACY_CACHE_PURGE:-false}"

NLTK_DATA="${NLTK_DATA:-/$GLADIA_PERSISTENT_PATH/nltk_data}"
NLTK_CACHE_PURGE="${NLTK_CACHE_PURGE:-false}"

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/$GLADIA_PERSISTENT_PATH/conda}"

for path in $PATH_TO_GLADIA_SRC $GLADIA_PERSISTENT_PATH $SPACY_CACHE_DIR $NLTK_DATA $MAMBA_ROOT_PREFIX; do
    if [ ! -d $path ]; then
        mkdir -p $path
    fi
done

cat tools/version/${GLADIA_VARIANT:-lite}.txt
echo build: $(cat tools/version/build)

P="\e[35m"
C="\e[36m"
G="\e[32m"
R="\e[31m"
EC="\e[0m"
echo -e "${P}== INIT Micromamba Server Env ==${EC}"
if [ -f $MAMBA_ROOT_PREFIX/envs/server/server.yml ]; then
    echo -e "${C}Updating micromamba server env.${EC}"
    micromamba update -f $PATH_TO_GLADIA_SRC/env.yaml -a
    cp $PATH_TO_GLADIA_SRC/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]==0.3.25\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
else
    echo -e "${G}Creating micromamba server.${EC}"
    micromamba create -f env.yaml -y
    cp $PATH_TO_GLADIA_SRC/env.yaml $MAMBA_ROOT_PREFIX/envs/server/server.yml
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]==0.3.25\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    micromamba run -n server /bin/bash -c "pip install \"jax[cuda11_cudnn82]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
fi

export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

echo -e "${P}== INIT Micromamba Venvs ==${EC}"
micromamba run -n server --cwd $VENV_BUILDER_PATH /bin/bash -c "python3 create_custom_envs.py --modality '.*' --debug_mode";

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/"

echo -e "${P}== FORCE pip upgrade in virtualenvs ==${EC}"
# Not optimal since we run this everytime, even after afresh install
for d in ${MAMBA_ROOT_PREFIX}/envs/*/; do
    cd $d
    echo "${C}Updating $(basename $d) pip packages.${EC}"
    micromamba activate $(basename $d)
    if [ -f "./$(basename $d).yaml" ]; then csplit -s --suppress-matched $d/$(basename $d).yaml '/- pip:/' '{*}'; fi
    if [ -f "./$(basename $d).yml" ]; then csplit -s --suppress-matched $d/$(basename $d).yml '/- pip:/' '{*}'; fi
    sed "s/ - //" xx01 > reqs.txt
    pip install --upgrade -r reqs.txt
done
micromamba activate server

echo -e "${P}== FIX Protobuh ==${EC}"
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O $MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/google/protobuf/internal/builder.py

echo -e "${P}== ADJUST path rights ==${EC}"
chown -R $DOCKER_USER:$DOCKER_GROUP $PATH_TO_GLADIA_SRC $GLADIA_TMP_PATH $GLADIA_TMP_PATH $GLADIA_PERSISTENT_PATH

echo -e "${P}== FIX libcurl references ==${EC}"
rm $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.6.0 $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
$CLEAN_LAYER_SCRIPT

echo -e "${P}== START supervisor ==${EC}"
service supervisor start

echo -e "${P}== INIT nltk + Spacy ==${EC}"
if [ "$SPACY_CACHE_PURGE" == "true" ]; then
    echo -e "${R}Purging Spacy cache.${EC}"
    rm -rvf $SPACY_CACHE_DIR
fi
if [ "$NLTK_CACHE_PURGE" == "true" ]; then
    echo -e "${R}Purging NLTK cache.${EC}"
    rm -rvf $NLTK_DATA
fi
micromamba run -n server --cwd $PATH_TO_GLADIA_SRC python prepare.py

echo -e "${P}== START Gladia ==${EC}"
micromamba run -n server --cwd $PATH_TO_GLADIA_SRC gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP:-8080} \
--workers ${API_SERVER_WORKERS:-1} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT:-1200}
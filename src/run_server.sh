#!/bin/bash
MODE="${MODE:-standalone}"

PATH_TO_GLADIA_SRC="${PATH_TO_GLADIA_SRC:-/app}"

GLADIA_PERSISTENT_PATH="${GLADIA_PERSISTENT_PATH:-/gladia}"

SPACY_CACHE_DIR="${SPACY_CACHE_DIR:-$GLADIA_PERSISTENT_PATH/spacy/models}"
SPACY_CACHE_PURGE="${SPACY_CACHE_PURGE:-false}"

NLTK_DATA="${NLTK_DATA:-/$GLADIA_PERSISTENT_PATH/nltk_data}"
NLTK_CACHE_PURGE="${NLTK_CACHE_PURGE:-false}"

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/$GLADIA_PERSISTENT_PATH/conda}"

# if MANUAL_FORCE_UPDATE is true, skip the update
# this is useful for devs for faster server start
# its a flag from the bash
# Initialize the flag variable
MANUAL_SKIP_UPDATE=false
MANUAL_SKIP_PREPARE=false

help_message="Usage: ./run_server.sh [-f] [-s] [-h] [-p]

Options:
  -s  Set the flag to skip manually server env update
  -v  Set the flag to skip manually venvs updates
  -p  Set the flag to skip manually gladia preparation's steps
  -h  Display this help message"

# Use getopts to parse the options
while getopts ":svph" opt; do
  case $opt in
    s) MANUAL_SKIP_SERVER_UPDATE=$OPTARG;;
    v) MANUAL_SKIP_VENV_UPDATE=$OPTARG;;
    p) MANUAL_SKIP_PREPARE=$OPTARG;;
    h) echo "$help_message"
       exit 0;;
    \?) echo "Invalid option: -$OPTARG" >&2
        exit 1;;
  esac
done

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

echo -e "${C}Checking micromamba minimal server env requirements.${EC}"
if micromamba env list | grep envs/server; then
  echo -e "${C}Server env exists.${EC}"
else
  echo -e "${C}Server env doesn't exists.${EC}"
  echo -e "${C}Creating server env.${EC}"
  micromamba create -n server python=3.8 -y

  echo -e "${C}Installing minimal requirements.${EC}"
  micromamba -n server install conda-forge::pyyaml conda-forge::tqdm
fi

# if MANUAL_SKIP_UPDATE is set to true, skip the update
# this is useful for devs for faster server start
# its a flag from the bash
if [ "$MANUAL_SKIP_SERVER_UPDATE" == "true" ]; then
  echo -e "${C}Skipping Server env update manually .${EC}"
else
  echo -e "${C}Updating Server env.${EC}"
  micromamba run -n server --cwd $VENV_BUILDER_PATH /bin/bash -c "python3 create_custom_envs.py  --server_env --debug_mode --debug_mode --force_recreate --python_version=3.8"
fi

if [ "$MANUAL_SKIP_VENV_UPDATE" == "true" ]; then
  echo -e "${C}Skipping Venvs update manually .${EC}"
else
  echo -e "${P}== INIT Micromamba Venvs if needed ==${EC}"
  micromamba run -n server --cwd $VENV_BUILDER_PATH /bin/bash -c "python3 create_custom_envs.py --modality '.*' --debug_mode --python_version=3.8";
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/"

# Check if the string is already present in .bashrc
# If not, add it
STRING_EXISTS=`grep "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/\"" ~/.bashrc`

if [ -z "$STRING_EXISTS" ]; then
    # String is not present, add it
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/\"" >> ~/.bashrc
fi

# same for cublas
STRING_EXISTS=`grep "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/\"" ~/.bashrc`
if [ -z "$STRING_EXISTS" ]; then
    # String is not present, add it
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/nvidia/cublas/lib/\"" >> ~/.bashrc
fi

echo -e "${P}== FIX Protobuf ==${EC}"
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O $MAMBA_ROOT_PREFIX/envs/server/lib/python3.8/site-packages/google/protobuf/internal/builder.py

echo -e "${P}== ADJUST path rights ==${EC}"
chown -R $DOCKER_USER:$DOCKER_GROUP $PATH_TO_GLADIA_SRC $GLADIA_TMP_PATH $GLADIA_TMP_PATH $GLADIA_PERSISTENT_PATH

echo -e "${P}== FIX libcurl references ==${EC}"
rm $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.6.0 $MAMBA_ROOT_PREFIX/envs/server/lib/libcurl.so.4
$CLEAN_LAYER_SCRIPT

echo -e "${P}== START supervisor ==${EC}"
service supervisor start
echo
echo -e "${P}== INIT nltk + Spacy ==${EC}"
if [ "$SPACY_CACHE_PURGE" == "true" ]; then
    echo -e "${R}Purging Spacy cache.${EC}"
    rm -rvf $SPACY_CACHE_DIR
fi
if [ "$NLTK_CACHE_PURGE" == "true" ]; then
    echo -e "${R}Purging NLTK cache.${EC}"
    rm -rvf $NLTK_DATA
fi

#if MANUAL_SKIP_PREPARE is set to true, skip the prepare
# this is useful for devs for faster server start
# its a flag from the bash
if [ "$MANUAL_SKIP_PREPARE" == "true" ]; then
  echo -e "${C}Skipping prepare manually .${EC}"
else
  echo -e "${P}== PREPARE Gladia ==${EC}"
  micromamba run -n server --cwd $PATH_TO_GLADIA_SRC python prepare.py
fi

echo -e "${P}== START Gladia ==${EC}"
micromamba run -n server --cwd $PATH_TO_GLADIA_SRC gunicorn main:app \
-b 0.0.0.0:${API_SERVER_PORT_HTTP:-8080} \
--workers ${API_SERVER_WORKERS:-1} \
--worker-class uvicorn.workers.UvicornWorker \
--timeout ${API_SERVER_TIMEOUT:-1200}
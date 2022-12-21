#https://www.docker.com/blog/advanced-dockerfiles-faster-builds-and-smaller-images-using-buildkit-and-multistage-builds/
ARG GLADIA_DOCKER_BASE=nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04


FROM $GLADIA_DOCKER_BASE

ARG SKIP_CUSTOM_ENV_BUILD="false"
ARG SKIP_ROOT_CACHE_CLEANING="false"
ARG SKIP_PIP_CACHE_CLEANING="false"
ARG SKIP_YARN_CACHE_CLEANING="false"
ARG SKIP_NPM_CACHE_CLEANING="false"
ARG SKIP_TMPFILES_CACHE_CLEANING="false"
ARG GLADIA_TMP_PATH="/tmp/gladia"
ARG GLADIA_PERSISTENT_PATH="/gladia"
ARG PATH_TO_GLADIA_SRC="/app"
ARG DOCKER_USER=root
ARG DOCKER_GROUP=root
ARG API_SERVER_PORT_HTTP=8080
ARG MAMBA_ALWAYS_SOFTLINK="true"
ARG CLEAN_LAYER_SCRIPT=$PATH_TO_GLADIA_SRC/tools/docker/clean-layer.sh
ARG VENV_BUILDER_PATH=$PATH_TO_GLADIA_SRC/tools/venv-builder/
ARG GLADIA_BUILD="unknown"

ENV DOCKER_USER=$DOCKER_USER \
    DOCKER_GROUP=$DOCKER_GROUP \
    PATH_TO_GLADIA_SRC=$PATH_TO_GLADIA_SRC \
    VENV_BUILDER_PATH=$VENV_BUILDER_PATH \
    GLADIA_PERSISTENT_PATH=$GLADIA_PERSISTENT_PATH \
    GLADIA_MODEL_PATH=$GLADIA_PERSISTENT_PATH/model \
    TRANSFORMERS_CACHE=$GLADIA_MODEL_PATH/transformers \
    TORCH_HOME=$GLADIA_MODEL_PATH/torch/hub \
    PYTORCH_TRANSFORMERS_CACHE=$GLADIA_MODEL_PATH/pytorch_transformers \
    PYTORCH_PRETRAINED_BERT_CACHE=$GLADIA_MODEL_PATH/pytorch_pretrained_bert \
    TORCH_HUB=$GLADIA_MODEL_PATH/torch/hub \
    MII_CACHE_PATH=$GLADIA_MODEL_PATH/mii/cache \
    MII_MODEL_PATH=$GLADIA_MODEL_PATH/mii/models \
    NLTK_DATA=$GLADIA_PERSISTENT_PATH/nltk \
    PIPENV_VENV_IN_PROJECT="enabled" \
    TOKENIZERS_PARALLELISM="true" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    distro="ubuntu2004" \
    arch="x86_64" \
    TZ="UTC" \
    MAMBA_ROOT_PREFIX=$GLADIA_PERSISTENT_PATH/conda \
    MAMBA_EXE="/usr/local/bin/micromamba" \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ALWAYS_YES=true \
    PATH=$PATH:/usr/local/bin/:$MAMBA_EXE \
    API_SERVER_WORKERS=1 \
    API_SERVER_PORT_HTTP=$API_SERVER_PORT_HTTP \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    JAX_PLATFORM_NAME=gpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN mkdir -p $GLADIA_TMP_PATH \
             $GLADIA_PERSISTENT_PATH \
             $GLADIA_MODEL_PATH \
             $TRANSFORMERS_CACHE \
             $PYTORCH_TRANSFORMERS_CACHE \
             $PYTORCH_PRETRAINED_BERT_CACHE \
             $NLTK_DATA \
             $PATH_TO_GLADIA_SRC \
             $MII_CACHE_PATH \
             $MII_MODEL_PATH

COPY ./tools/docker/clean-layer.sh $CLEAN_LAYER_SCRIPT

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
# Update apt repositories - Add Nvidia GPG key
    apt-key del 7fa2af80 && \
    apt-get update && \
    apt-get install -y apt-transport-https software-properties-common wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update --allow-insecure-repositories -y && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt update && \
    apt install kitware-archive-keyring && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt update && \
    apt install -y \
        curl \
        twine \
        unzip \
        libssl-dev \
        libpng-dev \
        libjpeg-dev \
        python3.8 \
        python3.8-distutils \
        python3.8-dev \
        python3-setuptools \
        git-lfs \
        libmagic1 \
        libmysqlclient-dev \
        libgl1 \
        software-properties-common \
        cmake \
        supervisor \
        libleptonica-dev \
        tesseract-ocr  \
        libtesseract-dev \
        python3-pil \
        tesseract-ocr-all \
        poppler-utils \
        imagemagick \
        libsndfile1 \
        ffmpeg \
        nvidia-cuda-toolkit \
        protobuf-compiler && \
    #echo "== ADJUSTING binaries ==" && \
    #ln -sf /usr/bin/python38 /usr/bin/python3 && \
    #ln -sf /usr/bin/python /usr/bin/python3 && \
    echo "== INSTALLING GITLFS ==" && \
    cd /tmp && \
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-386-v3.0.1.tar.gz && \
    tar -xvf git-lfs-linux-386-v3.0.1.tar.gz && \
    bash /tmp/install.sh && \
    rm /tmp/install.sh && \
    echo "== INSTALLING CUDNN ==" && \
    wget "https://storage.gra.cloud.ovh.net/v1/AUTH_90df0bdc74f749ce86783e6550b1e4aa/filehosting/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz" && \
    tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH && \
    rm -rf cudnn* && \
    echo "== INSTALLING UMAMBA ==" && \
    wget -qO- "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xvj bin/micromamba && \
    mv bin/micromamba /usr/local/bin/micromamba && \
    micromamba shell init -s bash && \
    micromamba config set always_softlink $MAMBA_ALWAYS_SOFTLINK && \
    git clone https://github.com/Tencent/rapidjson.git && \
    cd rapidjson && \
    cmake . && \
    make && \
    make install && \
    $CLEAN_LAYER_SCRIPT

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# TODO: this deepspeed fix should be implemented at the deepspeed
# level later with a PR
# here: https://github.com/microsoft/DeepSpeed-MII/blob/cd6a07f6f6616d2378b3e05c90ec7ba234b888f7/mii/deployment.py#L97
RUN ln -s $MII_MODEL_PATH /tmp/mii_models

COPY . $PATH_TO_GLADIA_SRC

# Automatically activate micromaba for every bash shell
RUN mv $PATH_TO_GLADIA_SRC/tools/docker/_activate_current_env.sh /usr/local/bin/ && \
    echo "source /usr/local/bin/_activate_current_env.sh" >> ~/.bashrc && \
    echo "source /usr/local/bin/_activate_current_env.sh" >> /etc/skel/.bashrc && \
    echo "micromamba activate server" >> ~/.bashrc

RUN echo $GLADIA_BUILD > $PATH_TO_GLADIA_SRC/tools/version/build

WORKDIR $PATH_TO_GLADIA_SRC

ENTRYPOINT ["micromamba", "run", "-n", "server"]

CMD ["run_server.sh"]
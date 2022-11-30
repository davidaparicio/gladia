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
ARG GLADIA_TMP_MODEL_PATH=$GLADIA_TMP_PATH/models
ARG PATH_TO_GLADIA_SRC="/app"
ARG DOCKER_USER=root
ARG DOCKER_GROUP=root
ARG API_SERVER_PORT_HTTP=8080
ARG MAMBA_ALWAYS_SOFTLINK="true"
ARG CLEAN_LAYER_SCRIPT=$PATH_TO_GLADIA_SRC/tools/docker/clean-layer.sh
ARG VENV_BUILDER_PATH=$PATH_TO_GLADIA_SRC/tools/venv-builder

ENV GLADIA_TMP_MODEL_PATH=$GLADIA_TMP_MODEL_PATH

ENV TRANSFORMERS_CACHE=$GLADIA_TMP_MODEL_PATH/transformers \
    PIPENV_VENV_IN_PROJECT="enabled" \
    PYTORCH_TRANSFORMERS_CACHE=$GLADIA_TMP_MODEL_PATH/pytorch_transformers \
    PYTORCH_PRETRAINED_BERT_CACHE=$GLADIA_TMP_MODEL_PATH/pytorch_pretrained_bert \
    NLTK_DATA=$GLADIA_TMP_MODEL_PATH/nltk \
    TOKENIZERS_PARALLELISM="true" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    distro="ubuntu2004" \
    arch="x86_64" \
    PATH_TO_GLADIA_SRC="/app" \
    API_SERVER_WORKERS=1 \
    TORCH_HOME=$GLADIA_TMP_MODEL_PATH/torch/hub \
    MAMBA_ROOT_PREFIX="/opt/conda" \
    MAMBA_EXE="/usr/local/bin/micromamba" \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ALWAYS_YES=true \
    PATH=$PATH:/usr/local/bin/:$MAMBA_EXE \
    PATH_TO_GLADIA_SRC=$PATH_TO_GLADIA_SRC \
    API_SERVER_PORT_HTTP=$API_SERVER_PORT_HTTP

RUN mkdir -p $GLADIA_TMP_PATH && \
    mkdir -p $TRANSFORMERS_CACHE && \
    mkdir -p $PYTORCH_TRANSFORMERS_CACHE && \
    mkdir -p $PYTORCH_PRETRAINED_BERT_CACHE && \
    mkdir -p $NLTK_DATA && \
    mkdir -p $PATH_TO_GLADIA_SRC

ADD ./tools/docker/clean-layer.sh $CLEAN_LAYER_SCRIPT

ENV TZ="UTC"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    mkdir -p $GLADIA_TMP_PATH && \
    mkdir -p $TRANSFORMERS_CACHE && \
    mkdir -p $PYTORCH_TRANSFORMERS_CACHE && \
    mkdir -p $PYTORCH_PRETRAINED_BERT_CACHE && \
    mkdir -p $NLTK_DATA && \
    mkdir -p $PATH_TO_GLADIA_SRC && \
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
    echo "== ADJUSTING binaries ==" && \
    mv /usr/bin/python3 /usr/bin/python38 && \
    ln -sf /usr/bin/python /usr/bin/python3 && \
    echo "== INSTALLING GITLFS ==" && \
    cd /tmp && \
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-386-v3.0.1.tar.gz && \
    tar -xvf git-lfs-linux-386-v3.0.1.tar.gz && \
    bash /tmp/install.sh && \
    rm /tmp/install.sh && \
    echo "== INSTALLING CUDNN ==" && \
    wget -qO- "https://storage.gra.cloud.ovh.net/v1/AUTH_90df0bdc74f749ce86783e6550b1e4aa/filehosting/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz" | tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH && \
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

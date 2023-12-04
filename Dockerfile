FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

ARG MODEL_PATH="liuhaotian/llava-v1.5-7b"
ARG MODEL_NAME="llava-v1.5-7b"
ARG LOAD_4BIT="false"
ARG LOAD_8BIT="false"

ENV MODEL_PATH=${MODEL_PATH} \
    MODEL_NAME=${MODEL_NAME} \
    LOAD_4BIT=${LOAD_4BIT} \
    LOAD_8BIT=${LOAD_8BIT}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

WORKDIR /

RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    python3.10-venv \
    python3-pip \
    python3-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    ln -s /usr/bin/python3.10 /usr/bin/python

RUN python3 -m venv /venv

RUN source /venv/bin/activate && \
    pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==0.0.22 && \
    deactivate

WORKDIR /
COPY . /LLaVA

WORKDIR /LLaVA
RUN source /venv/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install wheel && \
    pip3 install -e . && \
    pip3 install ninja && \
    pip3 install flash-attn --no-build-isolation && \
    pip3 install transformers==4.34.1 && \
    pip3 install flask && \
    pip3 install protobuf && \
    deactivate

RUN source /venv/bin/activate && \
    python3 scripts/download_model.py && \
    deactivate

ENV MODEL_PATH=/model_cache/${MODEL_NAME}

CMD [ "/venv/bin/python3", "-m", "llava.serve.api", "-H", "0.0.0.0", "-p", "8080" ]


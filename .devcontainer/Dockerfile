FROM nvidia/cuda:12.1.1-base-ubuntu20.04

# System packages (mirrors the project Dockerfile + dev tools)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gnupg2 \
        python3-pip \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git sox python3.8-dev python3.8-distutils qt5-default \
        curl vim less build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-install heavy Python deps so rebuilds are fast
RUN pip install torch torchvision torchaudio

# Pre-install project requirements (layer is cached until requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace

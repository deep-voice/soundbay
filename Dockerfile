FROM nvidia/cuda:12.1.1-base-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gnupg2 \
        python3-pip \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \ 
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \ 
     git sox python3.8-dev python3.8-distutils qt5-default\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note - QT5 is installed earlier hin the previous RUN command (qt5-default) - make sure to remove it from the requirements file when building the image from scratch
COPY requirements.txt .
RUN pip install torch torchvision torchaudio
RUN pip install --upgrade pip && pip install -e .
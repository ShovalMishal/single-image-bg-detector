ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Set bash as the default shell
ENV CUDA_HOME=/usr/local/cuda
ENV SHELL=/bin/bash

# To fix GPG key error when running apt-get update (https://github.com/NVIDIA/nvidia-docker/issues/1631)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    apt-utils \
    vim \
    git 

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
COPY . /workspace
WORKDIR /workspace
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"


RUN pip install ipdb
RUN pip install numba==0.48

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

RUN pip install tensorboard && \
    pip install timm==0.3.2 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html install torchtext==0.9.1

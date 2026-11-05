#------------------------------------------------------------
# Dockerfile: PyTorch + TensorFlow for RTX 40/50 Series GPUs
# Base: Ubuntu 24.04, CUDA 12.8.1, cuDNN
# Python: 3.13 (via uv)
# PyTorch: CUDA 13.4 nightly (cu134)
# TensorFlow: 2.22 custom Blackwell build (sm_120)
# User: Non-root (ibaad)
#------------------------------------------------------------

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

LABEL maintainer="Muhammad Abad Hussain"
LABEL description="PyTorch and TensorFlow GPU environment for NVIDIA Blackwell GPUs"

ENV DEBIAN_FRONTEND=noninteractive

#------------------------------------------------------------
# System dependencies
#------------------------------------------------------------

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        git \
        build-essential \
        pkg-config \
        libopenblas-dev \
        libjpeg-dev \
        libpng-dev \
        libhdf5-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------
# Create non-root user
#------------------------------------------------------------

ARG USERNAME=ibaad
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} \
            --gid ${USER_GID} \
            --create-home \
            --shell /bin/bash \
            ${USERNAME} && \
    mkdir -p /workspace && \
    chown -R ${USER_UID}:${USER_GID} /workspace

#------------------------------------------------------------
# Install uv and Python 3.13
#------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python

RUN uv venv --python 3.13 /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

#------------------------------------------------------------
# CUDA environment
#------------------------------------------------------------

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

#------------------------------------------------------------
# Install PyTorch CUDA 13.4 nightly
#------------------------------------------------------------

RUN uv pip install --python /opt/venv/bin/python \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu134

#------------------------------------------------------------
# Additional ML packages
#------------------------------------------------------------

RUN uv pip install --python /opt/venv/bin/python \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyter \
    jupyterlab \
    tqdm \
    pillow \
    h5py \
    pyyaml \
    tensorboard

#------------------------------------------------------------
# Install custom TensorFlow 2.22 Blackwell build
#------------------------------------------------------------

RUN uv pip install --python /opt/venv/bin/python --no-cache \
    "https://github.com/chivitiH/tensorflow-blackwell-python313/releases/download/v2.22.0-selfbuilt/tensorflow-2.22.0.dev0%2Bselfbuilt-cp313-cp313-linux_x86_64.whl"

#------------------------------------------------------------
# Copy utilities
#------------------------------------------------------------

COPY test_gpu.py /workspace/test_gpu.py
COPY startup.sh /startup.sh

RUN chmod +x /startup.sh && \
    chown -R ${USER_UID}:${USER_GID} /workspace

#------------------------------------------------------------
# Switch to non-root user
#------------------------------------------------------------

USER ibaad

WORKDIR /workspace

ENTRYPOINT ["/startup.sh"]
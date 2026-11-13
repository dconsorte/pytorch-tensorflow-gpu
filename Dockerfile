#------------------------------------------------------------
# Dockerfile: PyTorch + TensorFlow for RTX 40/50 Series GPUs
# Base: Ubuntu 24.04, CUDA 12.8.1 + cuDNN, Python 3.13 (via uv)
# PyTorch: stable cu128 build (includes Blackwell sm_120 kernels)
# TensorFlow: nightly by default; optional custom wheel via TF_WHEEL
#------------------------------------------------------------

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

LABEL maintainer="Dennis Consorte"
LABEL description="PyTorch and TensorFlow with GPU support for modern NVIDIA GPUs including Blackwell"

ENV DEBIAN_FRONTEND=noninteractive

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

# Install base dependencies (Python itself comes from uv below)
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
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (pinned) and create a Python 3.13 virtual environment
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy
RUN uv venv --python 3.13 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install PyTorch stable with CUDA 12.8 (native sm_120 support since 2.7)
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
RUN uv pip install --python /opt/venv/bin/python \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

# Install additional ML packages
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

# Install TensorFlow last so earlier installs can't overwrite its pinned dependencies.
# Default: tf-nightly from PyPI. To use a custom Blackwell build instead, pass a wheel URL:
#   docker build --build-arg TF_WHEEL=<url-or-path-to-wheel> -t pytorch-tensorflow-gpu .
# Check the license of any third-party wheel before using it.
ARG TF_WHEEL=""
RUN if [ -n "$TF_WHEEL" ]; then \
        uv pip install --python /opt/venv/bin/python "$TF_WHEEL"; \
    else \
        uv pip install --python /opt/venv/bin/python tf-nightly; \
    fi

# Copy utilities
COPY test_gpu.py /workspace/test_gpu.py
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

ENTRYPOINT ["/startup.sh"]

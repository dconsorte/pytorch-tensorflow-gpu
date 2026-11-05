#------------------------------------------------------------
# Dockerfile: PyTorch + TensorFlow for RTX 40/50 Series GPUs
# Base: Ubuntu 24.04 (container), CUDA 12.8, Python 3.13 (via uv)
# PyTorch: stable cu128 build (includes Blackwell sm_120 support)
# TensorFlow: 2.22 built natively for sm_120 (installed last)
#------------------------------------------------------------

FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu26.04

LABEL maintainer="Muhammad Abad Hussain"
LABEL description="PyTorch and TensorFlow with GPU support for modern NVIDIA GPUs including Blackwell validated for Ubuntu 26.04.1 LTS WSL2"

ENV DEBIAN_FRONTEND=noninteractive

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

# System dependencies (no PPA needed)
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

# Install uv, then use it to fetch Python 3.13 and create the venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv --python 3.13 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


RUN uv pip install --python /opt/venv/bin/python \
    torch \
    torchvision

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
    tensorboard \
    mlflow \
    dagshub \
    python-dotenv \
    dvc \
    dvc-s3

# Install TensorFlow 2.22 built natively for Blackwell (sm_120), Python 3.13
# Installed last so its dependencies aren't overwritten by the packages above
RUN uv pip install --python /opt/venv/bin/python --no-cache \
    "https://github.com/chivitiH/tensorflow-blackwell-python313/releases/download/v2.22.0-selfbuilt/tensorflow-2.22.0.dev0%2Bselfbuilt-cp313-cp313-linux_x86_64.whl"

# Copy utilities
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

ENTRYPOINT ["/startup.sh"]
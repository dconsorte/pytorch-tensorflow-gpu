#------------------------------------------------------------
# Dockerfile: PyTorch + TensorFlow for RTX 40/50 Series GPUs
# Base: Ubuntu 24.04, CUDA 12.8, Python 3.11
# Strategy: Use nightly builds with CUDA 12.8 for Blackwell (sm_120) support
#------------------------------------------------------------

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

LABEL maintainer="Dennis Consorte"
LABEL description="PyTorch and TensorFlow with GPU support for modern NVIDIA GPUs including Blackwell"

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

# Install base dependencies
RUN apt update --fix-missing && \
    apt install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
        ca-certificates \
        wget \
        curl \
        git \
        lsb-release && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        build-essential \
        pkg-config \
        libopenblas-dev \
        libjpeg-dev \
        libpng-dev \
        libhdf5-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip
RUN wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install PyTorch NIGHTLY with CUDA 12.8 for Blackwell support
# Using cu128 to match our CUDA runtime version
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128/

# Install TensorFlow nightly
RUN pip install tf-nightly

# Install additional ML packages
RUN pip install \
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

# Copy utilities
COPY test_gpu.py /workspace/test_gpu.py
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

ENTRYPOINT ["/startup.sh"]
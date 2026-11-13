# RTX 5090/5060 Docker: PyTorch & TensorFlow for NVIDIA Blackwell GPUs

**The first Linux Docker container fully tested and optimized for NVIDIA RTX 5090 and RTX 5060 Blackwell GPUs**, providing native support for both PyTorch and TensorFlow with CUDA 12.8. Run machine learning and deep learning workloads on the latest NVIDIA Blackwell architecture (RTX 50-series) with zero configuration on Ubuntu 24.04.2 LTS. Also supports RTX 40-series (4090, 4080, 4070) for mixed multi-GPU environments.

> **Fork Note**: This project is an enhanced fork of [wutzebaer/tensorflow-5090](https://github.com/wutzebaer/tensorflow-5090), extended to include PyTorch support, improved documentation, and additional utilities for multi-framework ML development.

## 📋 Table of Contents

- [Why This Container?](#-why-this-container)
- [Features](#-features)
- [Blackwell GPU Support Status](#-blackwell-gpu-support-status)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Using the Container](#-using-the-container)
- [GPU Control](#-gpu-control)
- [Common Tasks](#-common-tasks)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [What's Included](#-whats-included)
- [Author](#-author)

---

## 🎯 Why This Container?

If you've just bought an RTX 5090, RTX 5060, or other Blackwell GPU and discovered that:
- ❌ PyTorch doesn't work out-of-the-box (kernel errors)
- ❌ TensorFlow requires manual compilation
- ❌ Existing Docker images don't support SM 12.0 architecture
- ❌ Mixed RTX 50/40 series setups are complicated

**This container solves all these problems.** Get your Blackwell GPU running with PyTorch and TensorFlow in minutes, not days.

---

## 🚀 Features

- ✅ **RTX 5090 & RTX 5060 Tested** - Fully validated on actual Blackwell hardware
- ✅ **NVIDIA Blackwell Support** - SM 12.0 architecture with CUDA 12.8
- ✅ **PyTorch Stable (CUDA 12.8)** - Native Blackwell (sm_120) kernels, no nightly builds needed
- ✅ **TensorFlow Nightly** - JIT compilation for optimal Blackwell performance
- ✅ **Multi-GPU Ready** - Mix RTX 50-series with RTX 40-series (4090, 4080, 4070)
- ✅ **Zero Configuration** - Works out-of-the-box with RTX 5090/5060/5080/5070
- ✅ **Ubuntu 24.04 + Python 3.13** - Modern stack for latest hardware, Python managed by uv
- ✅ **Jupyter Lab Included** - Start training immediately

## 🎯 Blackwell GPU Support Status

### Tested Hardware
- **RTX 5090** ✅ Fully tested and operational
- **RTX 5060** ✅ Fully tested and operational
- **RTX 5080/5070** ⚡ Compatible (same architecture)

### Framework Compatibility
| Framework | Blackwell Support | Status |
|-----------|------------------|---------|
| **TensorFlow** | Full support via JIT | ✅ Production ready |
| **PyTorch** | Native sm_120 kernels (stable cu128) | ✅ Production ready |
| **CUDA/cuDNN** | Native 12.8 support | ✅ Full compatibility |

> **Note**: First-run TensorFlow operations compile PTX to native Blackwell code (one-time ~30s delay). PyTorch uses pre-compiled kernels where available and falls back to JIT compilation.

---

## 🏃 Quick Start

For users with RTX 5090/5060 or other Blackwell GPUs who already have Docker and NVIDIA drivers:

```bash
# Clone the repository
git clone https://github.com/dconsorte/pytorch-tensorflow-gpu.git
cd pytorch-tensorflow-gpu

# Create workspace
mkdir -p ~/ai_training_env

# Build and run
chmod +x build.sh
./build.sh
```

---

## 🎮 Prerequisites

### 1. NVIDIA GPU Driver (Required)

**⚠️ IMPORTANT: Docker containers cannot install GPU drivers - they must be installed on your host system first!**

#### Check if you have drivers installed:
```bash
nvidia-smi
```

If this command works and shows your GPU, skip to step 2. If not, install drivers:

#### Ubuntu Desktop (Easiest):
1. Open **Software & Updates**
2. Go to **Additional Drivers** tab
3. Select **NVIDIA driver metapackage from nvidia-driver-570-open (proprietary, tested)**
   - The "open kernel" version is recommended and tested
   - Version 570 is tested with RTX 50-series
4. Click **Apply Changes** and restart

#### Ubuntu Server/Terminal:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Restart
sudo reboot

# Verify after restart
nvidia-smi
```

### 2. Docker Installation

**Why Docker Buildx?** Modern Docker installations include Buildx by default, but some older or minimal installations might not have it. Our installation script ensures both Docker and Buildx are properly installed.

**Learn more about Docker:**
- [Get Started with Docker](https://docs.docker.com/get-started/)
- [Docker Buildx Documentation](https://docs.docker.com/reference/cli/docker/buildx/)

#### Check if Docker is installed:
```bash
docker --version
```

If Docker is not installed or you're unsure about Buildx:
```bash
# Our script handles everything - Docker Engine, Buildx, and NVIDIA Container Toolkit
chmod +x install_docker_and_buildx.sh
./install_docker_and_buildx.sh

# After installation, log out and back in, or run:
newgrp docker
```

---

## 💾 Installation

### Step 1: Clone the Repository

```bash
# Option A: Clone from GitHub
git clone https://github.com/dconsorte/pytorch-tensorflow-gpu.git
cd pytorch-tensorflow-gpu

# Option B: Download and extract ZIP
# Extract the downloaded ZIP and navigate to the folder
cd pytorch-tensorflow-gpu
```

### Step 2: Verify Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi  # Should show your GPU(s)

# Check Docker
docker run hello-world  # Should work without 'sudo'

# Check GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If any of these fail, go back to [Prerequisites](#-prerequisites).

### Step 3: Create Workspace

```bash
# This is where your projects and data will be stored
mkdir -p ~/ai_training_env
```

### Step 4: Build and Run

```bash
# Make build script executable
chmod +x build.sh

# Build and start container (15-30 minutes first time)
./build.sh
```

The script will:
1. Check your NVIDIA driver
2. Build the Docker image with all frameworks
3. Create and start the container
4. Mount your workspace at `/workspace` inside the container

---

## 🎮 Using the Container

### Starting and Stopping

```bash
# Start existing container (after first build)
docker start -ai PYTORCH_TENSORFLOW_GPU

# Stop container (preserves your work)
docker stop PYTORCH_TENSORFLOW_GPU

# Remove container (⚠️ WARNING: Only if rebuilding - this deletes the container!)
docker rm PYTORCH_TENSORFLOW_GPU
```

### First Time in Container

When you enter the container, you'll see a summary of installed versions. Test everything works:

```bash
# Run comprehensive GPU test
python /workspace/test_gpu.py

# Note: Warnings during the test are normal and can be ignored
# Look for the ✅ checkmarks at the end - these confirm everything is working

# Quick checks
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TF GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

### Jupyter Lab

```bash
# Inside container, start Jupyter
jupyter lab --ip=0.0.0.0 --no-browser --allow-root

# Access from browser at: http://localhost:8888
# Token will be shown in terminal
```

---

## 🎯 GPU Control

### Force CPU-Only Mode

Sometimes you want to use CPU only (debugging, comparison, GPU busy):

```python
# Method 1: Environment variable (before imports!)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import tensorflow as tf

# Method 2: PyTorch specific
device = torch.device('cpu')
model = model.to(device)

# Method 3: TensorFlow specific
tf.config.set_visible_devices([], 'GPU')
```

### Select Specific GPU

If you have multiple GPUs:

```python
# Use only GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use only GPU 1  
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Use GPUs 0 and 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
```

### Memory Management

```python
# Check GPU memory
def check_gpu_memory():
    import torch
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

# Clear GPU memory
def clear_gpu_memory():
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

# Auto-select GPU with enough memory
def auto_device(min_free_gb=5):
    import torch
    if not torch.cuda.is_available():
        return 'cpu'
    
    for i in range(torch.cuda.device_count()):
        free_gb = torch.cuda.mem_get_info(i)[0] / 1e9
        if free_gb >= min_free_gb:
            return f'cuda:{i}'
    
    return 'cpu'

# Usage
device = auto_device(min_free_gb=10)
model = model.to(device)
```

---

## 📝 Common Tasks

### Install Additional Packages

```bash
# Inside container
pip install transformers datasets accelerate

# For a specific project, consider using requirements.txt
pip install -r /workspace/your_project/requirements.txt
```

### Transfer Files

```bash
# Your ~/ai_training_env folder is mounted at /workspace
# Simply copy files to ~/ai_training_env on host

# From host:
cp mydata.csv ~/ai_training_env/

# Inside container:
ls /workspace/mydata.csv
```

### Save Your Work

All files in `/workspace` (container) are automatically saved to `~/ai_training_env` (host).

### Use with VS Code

1. Install "Remote - Containers" extension in VS Code
2. Open VS Code in your project folder
3. Click "Reopen in Container" when prompted
4. Select the running container

---

## 🔧 Troubleshooting

### RTX 5090/5060 Blackwell-Specific Issues

#### PyTorch "no kernel image" Error
The container ships stable PyTorch built for CUDA 12.8, which includes native sm_120 kernels, so this error should not appear. If it does, check which build is installed:
```
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# Expect a +cu128 version and 'sm_120' in the list
```

#### TensorFlow First-Run Compilation
Normal behavior for Blackwell GPUs:
```
# You'll see: "Running PTX compilation for sm_120"
# This happens once per operation type (~30 seconds)
# Subsequent runs are fast
```

#### Best Performance Tips for Blackwell
```python
# Enable TF32 for PyTorch (Blackwell optimized)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision (Blackwell has enhanced Tensor Cores)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### General Troubleshooting

You need to install NVIDIA drivers on your host system first. See [Prerequisites](#-prerequisites).

### "docker: command not found"

Run the Docker installation script:
```bash
./install_docker_and_buildx.sh
```

### "permission denied" for Docker commands

Either:
- Log out and back in after installation
- Run: `newgrp docker`
- Check you're in docker group: `groups | grep docker`

### Container already exists error

```bash
# Remove old container
docker stop PYTORCH_TENSORFLOW_GPU
docker rm PYTORCH_TENSORFLOW_GPU
# Run build.sh again
```

### GPU not detected in container

1. Check host GPU: `nvidia-smi`
2. Test Docker GPU: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
3. Ensure NVIDIA Container Toolkit is installed: `./install_docker_and_buildx.sh`

### PyTorch Blackwell (RTX 50-series) Issues

If you see:
```
no kernel image is available for execution on the device
```

PyTorch is probably a build without sm_120 kernels (for example, one installed from the default PyPI index inside the container). Reinstall from the CUDA 12.8 index:
```
uv pip install --python /opt/venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Out of Memory Errors

```python
# Clear memory before running
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Use smaller batch sizes
batch_size = 16  # Instead of 64

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Slow First Run with TensorFlow

TensorFlow compiles kernels on first use. You'll see PTX compilation messages - this is normal and only happens once.

---

## 📂 Project Structure

```
pytorch-tensorflow-gpu/
├── Dockerfile              # Container definition
├── build.sh               # Build and run script
├── test_gpu.py            # GPU testing utility
├── startup.sh             # Container startup script
├── install_docker_and_buildx.sh  # Docker installation helper
├── .dockerignore          # Build optimization
└── README.md              # This file
```

---

## 💻 System Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 4070 | RTX 5060 | RTX 5090 |
| **Architecture** | Ada Lovelace (SM 8.9) | Blackwell (SM 12.0) | Blackwell (SM 12.0) |
| **GPU Driver** | 525+ | 560+ | 560.35.03+ |
| **RAM** | 16GB | 32GB | 64GB+ |
| **Storage** | 30GB | 50GB | 100GB+ |
| **OS** | Ubuntu 20.04 | Ubuntu 22.04 | Ubuntu 24.04 |

### Supported GPUs
- **Blackwell (Primary)**: RTX 5090, RTX 5080, RTX 5070, RTX 5060
- **Ada Lovelace (Legacy)**: RTX 4090, RTX 4080, RTX 4070 Ti/Super, RTX 4070, RTX 4060 Ti, RTX 4060

---

## 📦 What's Included

| Component | Version | Blackwell Optimization |
|-----------|---------|------------------------|
| **Ubuntu** | 24.04 LTS | Latest kernel for GPU support |
| **CUDA** | 12.8.1 | Native Blackwell SM 12.0 |
| **cuDNN** | 9.x | Optimized for Blackwell |
| **Python** | 3.13 | uv-managed virtual environment |
| **PyTorch** | 2.11.0 (stable, cu128) | Native SM 12.0 kernels |
| **TensorFlow** | tf-nightly (latest) | JIT compilation for Blackwell |
| **Jupyter Lab** | Latest | Pre-configured |
| **Key Libraries** | Latest | NumPy, Pandas, Matplotlib, scikit-learn |

### Why These Builds?
- **PyTorch**: Stable releases built for CUDA 12.8 have included native Blackwell kernels since PyTorch 2.7, so nightly builds are no longer needed. Pin a different version with `--build-arg TORCH_VERSION=... --build-arg TORCHVISION_VERSION=... --build-arg TORCHAUDIO_VERSION=...`.
- **TensorFlow**: Nightly builds have the latest XLA optimizations for new architectures.
- **Python 3.13 via uv**: uv installs a standalone Python 3.13, so the image no longer depends on the deadsnakes PPA.

### Using a Custom TensorFlow Wheel
Some community TensorFlow builds compile sm_120 kernels directly, which can help if XLA compilation fails on your card with the nightly build. Pass the wheel's URL at build time:
```bash
docker build --build-arg TF_WHEEL=<wheel-url> -t pytorch-tensorflow-gpu .
```
The wheel must be built for Python 3.13 (`cp313`). Third-party wheels are not verified by this project, and some carry non-commercial licenses, so check the source and license before using one.

---

## 👤 Author

Created by **Dennis Consorte**  
🔗 [https://dennisconsorte.com](https://dennisconsorte.com)

---

## 📜 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

This project builds on [@wutzebaer](https://github.com/wutzebaer)'s pioneering work in [tensorflow-5090](https://github.com/wutzebaer/tensorflow-5090), which first enabled TensorFlow on RTX 5090 GPUs. This fork extends that foundation with:

- **Blackwell-Optimized PyTorch** - PyTorch built for CUDA 12.8 with native RTX 50-series support
- **Multi-GPU Support** - Mix Blackwell (RTX 5090/5060) with Ada Lovelace (RTX 4090)
- **Production Testing** - Validated on actual RTX 5090 and RTX 5060 hardware
- **Enhanced Documentation** - Specific guidance for Blackwell architecture
- **Automated Setup** - One-command installation for Blackwell GPU users
- **Performance Optimizations** - TF32 and mixed precision configs for Blackwell

---

## ⭐ Support

If this project helps you run ML workloads on your RTX 5090, RTX 5060, or other Blackwell GPUs, please consider giving it a star on GitHub!

---

## 🏷️ Keywords & Topics

`RTX 5090` `RTX 5060` `RTX 5080` `RTX 5070` `Blackwell GPU` `NVIDIA Blackwell` `PyTorch Blackwell` `TensorFlow Blackwell` `CUDA 12.8` `RTX 50-series` `RTX 5090 Docker` `RTX 5060 Docker` `Machine Learning` `Deep Learning` `GPU Docker` `RTX 4090` `Multi-GPU` `Blackwell Architecture` `SM 12.0`
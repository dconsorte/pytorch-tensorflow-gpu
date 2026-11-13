#!/bin/bash
# build.sh - Build script for PyTorch + TensorFlow GPU container

set -e

echo "🛠️ Building PyTorch + TensorFlow GPU Docker image..."
echo "📋 Build info:"
echo "   - Image name: pytorch-tensorflow-gpu"
echo "   - Container name: PYTORCH_TENSORFLOW_GPU"
echo "   - CUDA 12.8 base"
echo "   - Python 3.13 (uv-managed virtual environment)"
echo "   - PyTorch stable with CUDA 12.8 (native Blackwell support)"
echo "   - TensorFlow nightly"
echo ""

# Check NVIDIA driver
echo "🔍 Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   Driver version: $driver_version"
else
    echo "   ⚠️  nvidia-smi not found"
fi

# Create workspace if needed
if [ ! -d "$HOME/ai_training_env" ]; then
    echo "📁 Creating workspace directory: ~/ai_training_env"
    mkdir -p ~/ai_training_env
fi

echo ""
echo "🔨 Building Docker image..."
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    -t pytorch-tensorflow-gpu \
    .

echo ""
echo "✅ Build completed!"
echo ""

# Check for existing container
if docker ps -a --format "table {{.Names}}" | grep -q "^PYTORCH_TENSORFLOW_GPU$"; then
    echo "⚠️  Container 'PYTORCH_TENSORFLOW_GPU' already exists."
    read -p "Remove existing container? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing container..."
        docker stop PYTORCH_TENSORFLOW_GPU 2>/dev/null || true
        docker rm PYTORCH_TENSORFLOW_GPU
    else
        echo "❌ Aborting. Please manually handle the existing container."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting container..."
docker run -it --gpus all \
  --name PYTORCH_TENSORFLOW_GPU \
  --restart unless-stopped \
  -v ~/ai_training_env:/workspace \
  pytorch-tensorflow-gpu
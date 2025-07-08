#!/bin/bash
# Container startup script

echo "=== PyTorch + TensorFlow GPU Container ==="
echo "Image: pytorch-tensorflow-gpu"
echo "Container: PYTORCH_TENSORFLOW_GPU"
echo ""

# CUDA Version
echo -n "CUDA Runtime: "
nvcc --version | grep release | awk '{print $5}' | sed 's/,//' || echo "Not found"

# Python Version
echo -n "Python: "
python --version 2>&1

# PyTorch Version with nightly info
echo -n "PyTorch: "
python -c "import torch; v=torch.__version__; print(f'{v} (nightly)' if '+' in v else v)" 2>/dev/null || echo "Not installed"

# Check PyTorch CUDA support
echo -n "PyTorch CUDA: "
python -c "import torch; print(f'Yes (for {torch.cuda.device_count()} GPUs)' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "Error"

# TensorFlow Version
echo -n "TensorFlow: "
python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null || echo "Not installed"

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || echo "GPU detection failed"

echo ""
echo "Quick Commands:"
echo "  Test GPUs:            python /workspace/test_gpu.py"
echo "  Start Jupyter:        jupyter lab --ip=0.0.0.0 --allow-root"
echo "  Python shell:         python"
echo "==============================="
echo ""

# Check for Blackwell support
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        if cap[0] == 12:
            print('⚠️  Note: Blackwell GPUs detected (sm_120)')
            print('   PyTorch support is experimental')
            print('')
            print('   This container created by Dennis Consorte')
            print('   Licensed under MIT License')
            print('   https://dennisconsorte.com')
            print('   https://github.com/dconsorte')
            break
" 2>/dev/null

exec /bin/bash
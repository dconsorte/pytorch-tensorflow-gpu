#!/usr/bin/env python3
"""
GPU Test Script for TensorFlow and PyTorch
Tests CUDA availability and GPU compute capabilities

Author: Muhammad Abad Hussain
License: MIT
GitHub: https://github.com/dragonibaad
"""

import sys
import subprocess

def test_pytorch():
    """Test PyTorch GPU support"""
    print("=" * 60)
    print("🔥 PyTorch GPU Test")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n📊 GPU {i}: {props.name}")
                print(f"   Compute Capability: {props.major}.{props.minor}")
                print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"   SM Count: {props.multi_processor_count}")
                
                # Architecture detection
                if props.major == 8 and props.minor == 9:
                    print("   Architecture: Ada Lovelace (RTX 40-series)")
                elif props.major == 12 and props.minor == 0:
                    print("   Architecture: Blackwell (RTX 50-series)")
                
                # Quick computation test
                try:
                    x = torch.rand(1000, 1000, device=f'cuda:{i}')
                    y = torch.rand(1000, 1000, device=f'cuda:{i}')
                    z = torch.matmul(x, y)
                    torch.cuda.synchronize()
                    print(f"   ✅ Computation test passed")
                except Exception as e:
                    print(f"   ❌ Computation test failed: {e}")
        else:
            print("❌ No CUDA devices available")
            
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")

def test_tensorflow():
    """Test TensorFlow GPU support"""
    print("\n" + "=" * 60)
    print("🧠 TensorFlow GPU Test")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow Version: {tf.__version__}")
        
        # List GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ Number of GPUs: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"\n📊 GPU {i}: {gpu.name}")
                
                # Get detailed info
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Test computation
                    with tf.device(gpu.name):
                        a = tf.random.normal([1000, 1000])
                        b = tf.random.normal([1000, 1000])
                        c = tf.matmul(a, b)
                        print(f"   ✅ Computation test passed")
                except Exception as e:
                    print(f"   ❌ Computation test failed: {e}")
            
            # Show CUDA info
            print(f"\n✅ Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"✅ GPU Support: {tf.test.is_gpu_available()}")
            
        else:
            print("❌ No GPUs detected by TensorFlow")
            
    except ImportError:
        print("❌ TensorFlow not installed")
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")

def test_nvidia_smi():
    """Show nvidia-smi output"""
    print("\n" + "=" * 60)
    print("📊 NVIDIA System Management Interface")
    print("=" * 60)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ nvidia-smi command failed")
    except Exception as e:
        print(f"❌ Could not run nvidia-smi: {e}")

def test_cuda_compatibility():
    """Test CUDA compatibility with both frameworks"""
    print("\n" + "=" * 60)
    print("🔧 CUDA Compatibility Summary")
    print("=" * 60)
    
    try:
        import torch
        pytorch_cuda = torch.cuda.is_available()
        pytorch_version = torch.version.cuda if pytorch_cuda else "N/A"
    except:
        pytorch_cuda = False
        pytorch_version = "Not installed"
    
    try:
        import tensorflow as tf
        tf_cuda = tf.test.is_built_with_cuda()
        tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    except:
        tf_cuda = False
        tf_gpu = False
    
    print(f"PyTorch CUDA: {'✅' if pytorch_cuda else '❌'} (Version: {pytorch_version})")
    print(f"TensorFlow CUDA: {'✅' if tf_cuda else '❌'}")
    print(f"TensorFlow GPU: {'✅' if tf_gpu else '❌'}")
    
    # Architecture support summary
    print("\n Architecture Support:")
    print("- Ada Lovelace (SM 8.9): RTX 4090, 4080, 4070, 4060")
    print("- Blackwell (SM 12.0): RTX 5090, 5080, 5070, 5060")

if __name__ == "__main__":
    print("GPU Test Suite for AI/ML Frameworks")
    print("   Created by Muhammad Abad Hussain")
    print("   https://github.com/dragonibaad")
    print("=" * 60)
    
    test_pytorch()
    test_tensorflow()
    test_nvidia_smi()
    test_cuda_compatibility()
    
    print("\n Test suite completed!")
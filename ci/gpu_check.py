#!/usr/bin/env python3
"""Strict GPU check for CI: exits non-zero if PyTorch or TensorFlow can't compute on the GPU.

Run inside the built image, e.g.:
    docker run --rm -i --gpus all --entrypoint python pytorch-tensorflow-gpu - < ci/gpu_check.py
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

failures = []
warnings = []


def check(name, fn):
    try:
        detail = fn()
        print(f"PASS  {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


print(f"Python {sys.version.split()[0]}")

# ---------------- PyTorch ----------------
import torch  # noqa: E402

print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda})")


def torch_cuda():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return f"{torch.cuda.device_count()} device(s)"


def torch_arch():
    archs = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        want = f"sm_{major}{minor}"
        assert want in archs, f"{torch.cuda.get_device_name(i)} needs {want}, build has {archs}"
    return ", ".join(archs)


def torch_matmul():
    out = []
    for i in range(torch.cuda.device_count()):
        a = torch.randn(2048, 2048, device=f"cuda:{i}")
        b = torch.randn(2048, 2048, device=f"cuda:{i}")
        c = (a @ b).cpu()
        ref = a.cpu() @ b.cpu()
        err = (c - ref).abs().max().item()
        assert err < 1.0, f"GPU/CPU mismatch {err}"
        out.append(f"{torch.cuda.get_device_name(i)} ok")
    return "; ".join(out)


check("torch CUDA available", torch_cuda)
if torch.cuda.is_available():
    check("torch arch list covers GPU", torch_arch)
    check("torch matmul on GPU matches CPU", torch_matmul)

# ---------------- TensorFlow ----------------
import tensorflow as tf  # noqa: E402

print(f"TensorFlow {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:  # noqa: BLE001
        pass


def tf_gpus():
    assert gpus, "TensorFlow sees no GPUs"
    return f"{len(gpus)} device(s)"


def tf_matmul():
    with tf.device("/GPU:0"):
        a = tf.random.normal([2048, 2048])
        b = tf.random.normal([2048, 2048])
        c = tf.matmul(a, b)
    assert "GPU" in c.device, f"ran on {c.device}"
    return c.device


check("TF sees GPU", tf_gpus)
if gpus:
    check("TF matmul on GPU", tf_matmul)

    # XLA is reported but not fatal: stock TF builds may lack sm_120 XLA support.
    try:
        @tf.function(jit_compile=True)
        def f(x, y):
            return tf.nn.relu(tf.matmul(x, y))

        with tf.device("/GPU:0"):
            f(tf.random.normal([1024, 1024]), tf.random.normal([1024, 1024])).numpy()
        print("PASS  TF XLA (jit_compile) on GPU")
    except Exception as e:  # noqa: BLE001
        warnings.append("TF XLA")
        msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        print(f"WARN  TF XLA (jit_compile) on GPU: {type(e).__name__}: {msg}")

print()
print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
sys.exit(1 if failures else 0)

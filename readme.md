# TensorFlow for RTX 5070 (Custom Build)

This repository contains a custom-built TensorFlow wheel file optimized for NVIDIA RTX 5090 GPUs.

## System Requirements

- **Python**: 3.10.x
- **CUDA**: 12.8.1
- **cuDNN**: 9.8.0
- **GPU**: NVIDIA RTX 5070 TI 
- **OS**: Ubuntu 22.04 

## Installation Instructions

```bash
# Create a new directory for testing TensorFlow installation
mkdir -p ~/tensorflow_test

# Create a virtual environment with Python 3.10 (crucial to use the same Python version)
python3.10 -m venv ~/tensorflow_test/venv

# Activate the virtual environment
source ~/tensorflow_test/venv/bin/activate

# Verify Python version in the virtual environment
python --version
# Should output: Python 3.10.x

# Install the TensorFlow wheel file
pip install https://github.com/weyn9q/rtx5070tensorflow/releases/download/v1.0/tensorflow-2.20.0.dev0+selfbuilt-cp310-cp310-linux_x86_64.whl

# Test TensorFlow installation and GPU detection
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}'); print('GPU test:', tf.test.is_gpu_available())"

# Optional: Run a simple GPU computation test
python -c "import tensorflow as tf; print('Testing GPU computation...'); a = tf.random.normal([5000, 5000]); b = tf.random.normal([5000, 5000]); c = tf.matmul(a, b); print('Matrix multiplication completed successfully on GPU')"

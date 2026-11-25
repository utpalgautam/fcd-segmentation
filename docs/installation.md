# Installation & Environment Setup Guide

This guide details how to install and configure the FCD Lesion Segmentation pipeline on Linux, Windows, and Containerized GPU Servers (e.g. NVIDIA H200 / A100 via Docker).

---

## 1. Linux Workstation (Ubuntu / Debian)

### Prerequisites
- Python 3.11 or 3.12
- Git
- NVIDIA GPU with CUDA 12+ drivers (optional for CPU-only development, recommended for training)

```bash
# 1. Install system tools
sudo apt update
sudo apt install -y git python3-venv python3-pip

# 2. Clone the repository
git clone https://github.com/utpalkant12/fcd-segmentation.git
cd fcd-segmentation

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA support (adjust cu124/cu128 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. Install fcd-segmentation in editable mode
pip install -e ".[dev,imaging]"

# 6. Verify environment setup
python scripts/train.py --check
```

---

## 2. Windows 11 / 10 (PowerShell)

### Prerequisites
- Python 3.11 or 3.12 (from python.org with "Add Python to PATH" enabled)
- Git for Windows
- NVIDIA GPU with current drivers

```powershell
# 1. Clone the repository
git clone https://github.com/utpalkant12/fcd-segmentation.git
Set-Location fcd-segmentation

# 2. Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 3. Upgrade pip and install PyTorch with CUDA
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install fcd-segmentation in editable mode
pip install -e ".[dev,imaging]"

# 5. Verify environment setup
python scripts/train.py --check
```

---

## 3. High-Performance GPU Server / Cluster (NVIDIA Tesla H200, A100 via Docker)

For institutional servers running Docker and Multi-Instance GPU (MIG) slices:

```bash
# 1. Connect via SSH
ssh username@server_ip

# 2. Check available MIG GPU slices
nvidia-smi -L
# Copy a free MIG UUID (e.g. MIG-30161203-9f1d-51ad-aaab-3a670439d5e3)

# 3. Start a persistent tmux session
tmux new -s fcd_run

# 4. Launch the NVIDIA PyTorch NGC container
docker run -it --rm \
  --gpus device=<PASTE_MIG_ID_HERE> \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/username/fcd-segmentation:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.02-py3 /bin/bash

# 5. Inside container: install package in editable mode
pip install --no-cache-dir -e .

# 6. Verify CUDA availability
python3 -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 4. Environment Verification

Confirm that all system components, libraries, and nnU-Net path bindings are functional:

```bash
python scripts/train.py --check
```

Expected output:
```text
=== nnU-Net Environment & System Status ===
Python: 3.12.x
CUDA Available: True (Device: NVIDIA ...)
Packages:
  torch: >= 2.0.0
  nnunetv2: 2.8.x
Effective nnU-Net Environment Paths:
  $nnUNet_raw -> data/processed
  $nnUNet_preprocessed -> data/nnunet_preprocessed
  $nnUNet_results -> outputs/nnunet_results
==========================================
Workflow action 'check' finished. Status: SUCCESS
```

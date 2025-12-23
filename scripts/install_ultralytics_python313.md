# Installing ultralytics on Python 3.13

## Current Status (December 2024)
PyTorch does not yet have official pre-built wheels for Python 3.13. However, there are workarounds:

## Option 1: Use Python 3.12 (Recommended)

The easiest solution is to use Python 3.12, which has full PyTorch support:

```bash
# Install Python 3.12 via Homebrew (macOS)
brew install python@3.12

# Create virtual environment
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install ultralytics
pip install ultralytics opencv-python

# Install rest of SciTrans
pip install -e .
```

Or use the automated script:
```bash
./scripts/setup_python312_for_yolo.sh
```

## Option 2: Try PyTorch Nightly Build (Experimental)

PyTorch nightly builds sometimes have Python 3.13 support before official release:

```bash
# Try installing PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Then try ultralytics
pip install ultralytics
```

**Warning**: Nightly builds are unstable and may have bugs.

## Option 3: Build PyTorch from Source (Advanced)

This is complex and time-consuming, but possible:

```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Install dependencies
pip install -r requirements.txt

# Build (this takes a long time!)
python setup.py install
```

## Option 4: Use Conda (May Have Pre-built Wheels)

Conda sometimes has packages before PyPI:

```bash
conda create -n scitrans python=3.13
conda activate scitrans
conda install pytorch -c pytorch-nightly
pip install ultralytics
```

## Verification

After installation, verify it works:

```python
from ultralytics import YOLO
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test loading a model
model = YOLO('yolov8n.pt')  # Downloads automatically
print("✓ YOLO model loaded successfully")
```

## Recommendation

**Use Option 1 (Python 3.12)** - it's the most reliable and stable solution. Python 3.12 has all the features you need and full PyTorch/ultralytics support.


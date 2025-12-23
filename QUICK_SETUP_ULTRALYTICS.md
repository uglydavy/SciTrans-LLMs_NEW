# Quick Setup: Install ultralytics on Python 3.12

Since you need `ultralytics` and you're on Python 3.13 (which PyTorch doesn't support yet), here's the quickest solution:

## Step 1: Install Python 3.12

```bash
brew install python@3.12
```

## Step 2: Create Virtual Environment with Python 3.12

```bash
# Create new venv with Python 3.12
python3.12 -m venv .venv312

# Activate it
source .venv312/bin/activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

## Step 3: Install ultralytics

```bash
# Upgrade pip first
pip install --upgrade pip

# Install ultralytics (this will also install PyTorch)
pip install ultralytics opencv-python

# Verify installation
python -c "from ultralytics import YOLO; print('✓ ultralytics installed successfully')"
```

## Step 4: Install SciTrans Dependencies

```bash
# Install all SciTrans dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Step 5: Verify Everything Works

```bash
python -c "
from scitran.extraction.yolo import HAS_ULTRALYTICS, HAS_TORCH
from scitran.extraction.pdf_parser import PDFParser

print(f'YOLO available: {HAS_ULTRALYTICS}')
print(f'PyTorch available: {HAS_TORCH}')

# Test PDFParser with YOLO
parser = PDFParser(use_yolo=True)
print('✓ PDFParser with YOLO works!')
"
```

## Using the Automated Script

Or use the automated setup script:

```bash
./scripts/setup_python312_for_yolo.sh
```

This will:
1. Check for Python 3.12 (install if needed)
2. Create `.venv312` virtual environment
3. Install ultralytics and dependencies
4. Verify installation

## Switching Between Environments

**For YOLO features (Python 3.12):**
```bash
source .venv312/bin/activate
```

**For regular use without YOLO (Python 3.13):**
```bash
source .venv/bin/activate  # Your existing venv
```

## Alternative: Try PyTorch Nightly (Experimental)

If you want to stay on Python 3.13, you can try PyTorch nightly builds:

```bash
# In your Python 3.13 environment
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Then try ultralytics
pip install ultralytics
```

**Note**: Nightly builds may be unstable. Python 3.12 is more reliable.


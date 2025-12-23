#!/bin/bash
# Setup script to install Python 3.12 and ultralytics for YOLO support
# This is needed because PyTorch doesn't support Python 3.13 yet

set -e

echo "=========================================="
echo "Setting up Python 3.12 for ultralytics"
echo "=========================================="
echo ""

# Check if Python 3.12 is already available
if command -v python3.12 &> /dev/null; then
    echo "✓ Python 3.12 found: $(python3.12 --version)"
    PYTHON_CMD="python3.12"
else
    echo "Python 3.12 not found. Installing..."
    
    # Try Homebrew first (macOS)
    if command -v brew &> /dev/null; then
        echo "Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        PYTHON_CMD="python3.12"
    else
        echo "❌ Error: Python 3.12 not found and Homebrew not available."
        echo ""
        echo "Please install Python 3.12 manually:"
        echo "  1. macOS: brew install python@3.12"
        echo "  2. Or download from: https://www.python.org/downloads/"
        echo "  3. Then run this script again"
        exit 1
    fi
fi

echo ""
echo "Creating virtual environment with Python 3.12..."
VENV_DIR=".venv312"
$PYTHON_CMD -m venv $VENV_DIR

echo ""
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing ultralytics and dependencies..."
pip install ultralytics opencv-python

echo ""
echo "Verifying installation..."
python -c "from ultralytics import YOLO; import torch; print(f'✓ ultralytics: {YOLO.__version__ if hasattr(YOLO, \"__version__\") else \"OK\"}'); print(f'✓ PyTorch: {torch.__version__}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To install the rest of SciTrans dependencies:"
echo "  pip install -e ."
echo "  # Or: pip install -r requirements.txt"
echo ""


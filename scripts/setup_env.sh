#!/bin/bash
# Setup script for SciTrans-LLMs NEW
# This script creates a virtual environment and installs dependencies

set -e

echo "=========================================="
echo "  SciTrans-LLMs NEW - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first (minimal, guaranteed to work)
echo ""
echo "Installing core dependencies..."
pip install -r requirements-core.txt

# Install full dependencies (optional)
read -p "Install full dependencies including ML packages? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing full dependencies..."
    pip install -r requirements.txt
fi

# Install package in development mode
echo ""
echo "Installing scitran package..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import sys
print('Testing imports...')
try:
    from scitran import TranslationPipeline, Document, MaskingEngine
    print('✓ Core modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)

try:
    from scitran.extraction.pdf_parser import PDFParser
    print('✓ PDF parser available')
except ImportError:
    print('⚠ PyMuPDF not installed - PDF processing unavailable')

try:
    from scitran.translation.backends import CascadeBackend
    print('✓ Translation backends available')
except ImportError as e:
    print(f'⚠ Backend issue: {e}')

print('')
print('✅ Installation complete!')
"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the GUI:"
echo "  python gui/app.py"
echo ""
echo "To use the CLI:"
echo "  python -m cli.commands.main --help"
echo "  or: ./scitrans --help"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""


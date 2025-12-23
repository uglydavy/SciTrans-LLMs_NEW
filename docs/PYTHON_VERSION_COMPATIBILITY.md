# Python Version Compatibility

## Current Status

**SciTrans-LLMs** supports Python 3.9 through 3.13, but some optional dependencies have limitations:

### Core Features (All Python Versions)
- ✅ PDF parsing and extraction (PyMuPDF)
- ✅ Translation backends (OpenAI, Anthropic, DeepSeek, etc.)
- ✅ GUI interface (Gradio)
- ✅ All core translation features

### Optional ML Features (Python 3.9-3.12 Only)
- ⚠️ **PyTorch** (`torch`) - Required for:
  - HuggingFace translation backend
  - Advanced reranking features
  - YOLO layout detection (`ultralytics`)
  
**Current Limitation**: PyTorch does not yet provide pre-built wheels for Python 3.13. If you're using Python 3.13, these features will be unavailable until PyTorch adds support.

### Workarounds

#### Option 1: Use Python 3.11 or 3.12 (Recommended)
```bash
# Create a new virtual environment with Python 3.11 or 3.12
python3.11 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install ultralytics  # Now this will work
```

#### Option 2: Skip Optional ML Features (Python 3.13)
The system works perfectly fine without `ultralytics` and `torch`:
- Layout detection falls back to PyMuPDF + heuristic methods (still robust)
- Translation backends that don't require PyTorch work normally
- All core features remain available

Simply don't install `ultralytics` - the code handles its absence gracefully.

#### Option 3: Wait for PyTorch Support
PyTorch typically adds support for new Python versions within a few months. Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for updates.

## Checking Your Python Version

```bash
python3 --version
```

## Installing Without ML Dependencies

If you're using Python 3.13 and want to avoid dependency conflicts:

```bash
# Install core dependencies only
pip install -r requirements-minimal.txt

# Or install from pyproject.toml without ML extras
pip install -e .  # This won't install torch/ultralytics
```

## Verifying Installation

The system will automatically detect available features:

```python
from scitran.extraction.yolo import HAS_ULTRALYTICS, HAS_TORCH

print(f"YOLO available: {HAS_ULTRALYTICS}")
print(f"PyTorch available: {HAS_TORCH}")
```

If both are `False`, the system will use fallback methods (PyMuPDF + heuristics) which are still very robust for most documents.


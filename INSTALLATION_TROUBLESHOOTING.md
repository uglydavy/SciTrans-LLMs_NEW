# Installation Troubleshooting

## Issue: Cannot Install `ultralytics` on Python 3.13

### Problem
When trying to install `ultralytics`, you get:
```
ERROR: ResolutionImpossible: ... torch ... has no matching distributions available for your environment
```

### Root Cause
- You're using **Python 3.13**
- `ultralytics` requires `torch` (PyTorch)
- PyTorch doesn't yet provide pre-built wheels for Python 3.13
- This causes pip's dependency resolver to fail

### Solution Options

#### ✅ Option 1: Skip `ultralytics` (Recommended for Python 3.13)
**The system works perfectly fine without it!**

`ultralytics` is **optional** - it's only used for advanced YOLO-based layout detection. The system automatically falls back to PyMuPDF + heuristic detection, which is still very robust.

**Just don't install it:**
```bash
# Don't run: pip install ultralytics
# The system will work fine without it
```

**Verify it works:**
```bash
python3 -c "from scitran.extraction.pdf_parser import PDFParser; parser = PDFParser(); print('✓ Works!')"
```

#### Option 2: Use Python 3.11 or 3.12
If you specifically need YOLO layout detection:

```bash
# Install Python 3.11 or 3.12 (if not already installed)
# macOS: brew install python@3.11
# Or download from python.org

# Create new virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Now ultralytics will install successfully
pip install ultralytics
```

#### Option 3: Wait for PyTorch Support
PyTorch typically adds Python 3.13 support within a few months. Check:
- https://pytorch.org/get-started/locally/
- https://github.com/pytorch/pytorch/issues

### What Features Are Affected?

**Without `ultralytics`:**
- ❌ YOLO-based layout detection (advanced)
- ✅ PyMuPDF + heuristic layout detection (still robust)
- ✅ All translation features
- ✅ All PDF parsing features
- ✅ GUI interface
- ✅ All core functionality

**The system is designed to work without it!**

### Verification

Check what's available:
```python
from scitran.extraction.yolo import HAS_ULTRALYTICS, HAS_TORCH

print(f"YOLO available: {HAS_ULTRALYTICS}")
print(f"PyTorch available: {HAS_TORCH}")

if not HAS_ULTRALYTICS:
    print("✓ System will use PyMuPDF + heuristics (still robust)")
```

### Current Status

- **Python 3.13**: ✅ Core features work, ❌ YOLO unavailable (use Option 1)
- **Python 3.11/3.12**: ✅ Everything works including YOLO
- **Python 3.9/3.10**: ✅ Everything works including YOLO

### Quick Fix

**For Python 3.13 users:** Simply skip installing `ultralytics`. The system handles it gracefully and you won't notice any difference for most documents.


#!/bin/bash
# STEP 9: Clean repository script
# Removes unnecessary files and caches

echo "=== SciTrans Repository Cleanup ==="
echo

# Remove Python caches
echo "[1/5] Removing Python caches..."
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
echo "  ✓ Python caches removed"

# Remove test artifacts
echo "[2/5] Removing test artifacts..."
rm -rf artifacts/202* 2>/dev/null || true
rm -f test_translated.pdf test.pdf 2>/dev/null || true
rm -rf repro_output/ 2>/dev/null || true
rm -f *_overflow_report.json 2>/dev/null || true
echo "  ✓ Test artifacts removed"

# Remove temporary files
echo "[3/5] Removing temporary files..."
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ Temporary files removed"

# Remove large PDFs (optional - comment out if needed)
echo "[4/5] Removing large sample PDFs (keeping small test PDFs)..."
# Uncomment these lines if you want to remove large PDFs:
# rm -f alphafold.pdf attention_is_all_you_need.pdf clip_paper.pdf 2>/dev/null || true
# rm -f primes_in_arithmetic.pdf quantum_supremacy.pdf 2>/dev/null || true
echo "  ℹ  Skipped (uncomment in script to remove)"

# Remove empty directories
echo "[5/5] Removing empty directories..."
find . -type d -empty -not -path "./.git/*" -not -path "./.venv/*" -delete 2>/dev/null || true
echo "  ✓ Empty directories removed"

echo
echo "=== Cleanup Complete ==="
echo "Repository is now clean. Run 'git status' to see changes."


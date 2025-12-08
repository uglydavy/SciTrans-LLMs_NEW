#!/bin/bash
# Quick installation script for SciTrans-LLMs

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SciTrans-LLMs Quick Installation                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install in development mode
echo ""
echo "Installing SciTrans-LLMs..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install -e . 2>&1 | grep -E "(Successfully|Requirement already|error)" || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if installation succeeded
if command -v scitrans &> /dev/null; then
    echo "✓ Installation successful!"
    echo ""
    echo "Testing scitrans command..."
    scitrans --help > /dev/null 2>&1 && echo "✓ scitrans command works!" || echo "⚠ scitrans command has issues"
    echo ""
    
    # Test backends
    echo "Checking available backends..."
    scitrans backends 2>/dev/null || echo "Run: scitrans backends"
    echo ""
    
    # Quick test
    echo "Running quick test..."
    scitrans test --backend cascade --sample "Hello world" 2>&1 | grep -E "(Translation successful|Result:)" || echo "Test completed (check output above)"
    echo ""
    
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   🎉 INSTALLATION COMPLETE!                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  1. scitrans wizard          # Interactive mode"
    echo "  2. scitrans backends        # List all backends"
    echo "  3. scitrans translate file.pdf  # Translate a PDF"
    echo "  4. scitrans help            # Get detailed help"
    echo ""
else
    echo "⚠ Installation completed but 'scitrans' command not found in PATH"
    echo ""
    echo "Alternative: Use local script"
    echo "  chmod +x scitrans"
    echo "  ./scitrans --help"
    echo ""
fi

# Validate phases
if [ -f "validate_phases.py" ]; then
    echo "Run phase validation: python validate_phases.py"
fi

echo ""
echo "Documentation:"
echo "  • INSTALL.md          - Complete installation guide"
echo "  • READY_TO_USE.md     - Quick start guide"
echo "  • PHASE_VERIFICATION.md - Development phase status"
echo ""

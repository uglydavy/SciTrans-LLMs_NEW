#!/usr/bin/env python3
"""Quick verification script to check if GUI code has the latest fixes."""

import sys
from pathlib import Path

gui_file = Path(__file__).parent / "gui" / "app.py"

if not gui_file.exists():
    print("❌ gui/app.py not found!")
    sys.exit(1)

content = gui_file.read_text()

checks = {
    "__getattr__ method": "__getattr__" in content and "def __getattr__" in content,
    "_generate_translation_preview method": "def _generate_translation_preview" in content,
    "H0 entry log": 'translate_document:entry' in content,
    "Inline preview code": "COMPLETELY INLINE, NO METHOD DEPENDENCY" in content,
    "H16 before preview": "H16" in content and "before_preview" in content,
    "H17 preview start": "H17" in content and "preview_start" in content,
}

print("Verifying GUI code fixes...")
print("=" * 50)

all_ok = True
for name, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"{status} {name}")
    if not passed:
        all_ok = False

print("=" * 50)

if all_ok:
    print("\n✅ All fixes are in the code!")
    print("\n⚠️  IMPORTANT: The GUI process must be RESTARTED for changes to take effect.")
    print("   Steps:")
    print("   1. Stop the GUI (Ctrl+C in the terminal)")
    print("   2. Clear Python cache: find . -type d -name '__pycache__' -exec rm -rf {} +")
    print("   3. Restart: scitrans gui")
    print("\n   After restart, check the debug log for H0 entry log to confirm new code is running.")
    sys.exit(0)
else:
    print("\n❌ Some fixes are missing from the code!")
    sys.exit(1)



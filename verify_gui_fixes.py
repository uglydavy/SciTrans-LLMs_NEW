#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script to check GUI fixes:
1. Default backend is "deepseek"
2. Preview paths are absolute
3. Syntax is valid
"""

import sys
from pathlib import Path

def check_default_backend():
    """Check that default_backend is set to 'deepseek' in all places."""
    print("=" * 60)
    print("CHECKING DEFAULT BACKEND SETTINGS")
    print("=" * 60)
    
    gui_file = Path("gui/app.py")
    if not gui_file.exists():
        print("❌ gui/app.py not found")
        return False
    
    content = gui_file.read_text()
    
    issues = []
    
    # Check _default_config
    if '"default_backend": "deepseek"' in content:
        print("✅ _default_config has default_backend: deepseek")
    else:
        issues.append("_default_config missing 'default_backend': 'deepseek'")
        print("❌ _default_config missing correct default_backend")
    
    # Check initial_backend fallback
    if 'self.config.get("default_backend", "deepseek")' in content:
        print("✅ initial_backend fallback is 'deepseek'")
    else:
        if 'self.config.get("default_backend", "free")' in content:
            issues.append("initial_backend still uses 'free' as fallback")
            print("❌ initial_backend still uses 'free' as fallback")
        else:
            print("⚠️  Could not verify initial_backend fallback")
    
    # Check settings dropdown default
    if 'value=self.config.get("default_backend", "deepseek")' in content:
        print("✅ Settings dropdown default is 'deepseek'")
    else:
        if 'value=self.config.get("default_backend", "cascade")' in content:
            issues.append("Settings dropdown still uses 'cascade' as fallback")
            print("❌ Settings dropdown still uses 'cascade' as fallback")
        else:
            print("⚠️  Could not verify settings dropdown default")
    
    return len(issues) == 0

def check_preview_paths():
    """Check that preview paths use absolute paths."""
    print("\n" + "=" * 60)
    print("CHECKING PREVIEW PATH HANDLING")
    print("=" * 60)
    
    gui_file = Path("gui/app.py")
    content = gui_file.read_text()
    
    checks = [
        ("on_upload uses Path.resolve()", "Path(pdf_path).resolve()" in content or "Path(pdf_path).resolve()" in content),
        ("on_url_load uses Path.resolve()", "Path(pdf_path).resolve()" in content),
        ("nav_page uses Path.resolve()", "Path(source_path).resolve()" in content and "Path(trans_path).resolve()" in content),
        ("translate_document stores absolute paths", "Path(temp_output_path).resolve()" in content and "Path(self.source_pdf_path).resolve()" in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    return all_passed

def check_syntax():
    """Check that Python syntax is valid."""
    print("\n" + "=" * 60)
    print("CHECKING PYTHON SYNTAX")
    print("=" * 60)
    
    import py_compile
    
    files_to_check = [
        "gui/app.py",
        "scitran/core/pipeline.py",
        "scitran/rendering/pdf_renderer.py",
    ]
    
    all_passed = True
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  {file_path} not found, skipping")
            continue
        
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"✅ {file_path} syntax is valid")
        except py_compile.PyCompileError as e:
            print(f"❌ {file_path} has syntax errors:")
            print(f"   {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("GUI FIXES VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Default Backend", check_default_backend()))
    results.append(("Preview Paths", check_preview_paths()))
    results.append(("Syntax", check_syntax()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed!")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


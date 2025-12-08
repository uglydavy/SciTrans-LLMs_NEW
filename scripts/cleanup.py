#!/usr/bin/env python3
"""
Cleanup script for SciTrans-LLMs NEW

Removes redundant documentation files and organizes the project structure.

Usage:
    python scripts/cleanup.py          # Dry run (shows what would be deleted)
    python scripts/cleanup.py --execute  # Actually delete files
"""

import os
import sys
import argparse
from pathlib import Path

# Files to remove (redundant status/fix documentation)
REDUNDANT_FILES = [
    "ALL_CRITICAL_FIXES_COMPLETE.md",
    "ALL_ERRORS_FIXED.md",
    "ALL_ISSUES_RESOLVED.md",
    "ALL_WORKING_FINAL.md",
    "BUILD_SUMMARY.md",
    "COMPLETE_FIXES_AND_TESTING.md",
    "COMPLETE_GUIDE.md",
    "COMPLETE_SOLUTION.md",
    "CRITICAL_FIX_APPLIED.md",
    "ENHANCED_GUI_COMPLETE.md",
    "EVERYTHING_TESTED_AND_WORKING.md",
    "EVERYTHING_WORKING.md",
    "FINAL_COMPLETE_SUMMARY.md",
    "FINAL_COMPREHENSIVE_FIXES.md",
    "FINAL_FIXES_COMPLETE.md",
    "FINAL_STATUS.md",
    "FINAL_SUMMARY.md",
    "FIX_APPLIED.md",
    "GET_STARTED.md",
    "GUI_FIXED.md",
    "GUI_FIXES_APPLIED.md",
    "GUI_NOW_WORKING.md",
    "GUI_TROUBLESHOOTING.md",
    "IMPLEMENTATION_COMPLETE.md",
    "IMPLEMENTATION_STATUS.md",
    "INSTALL.md",
    "INSTALLATION_FIX.md",
    "INSTANT_DARK_MODE_WORKING.md",
    "LATEST_FIXES_SUMMARY.md",
    "NEW_FEATURES.md",
    "PHASE_VERIFICATION.md",
    "PROJECT_STRUCTURE.md",
    "READY_TO_USE.md",
    "UPDATES.md",
]

# Files to keep
KEEP_FILES = [
    "README.md",
    "QUICK_START.md",
    "API_KEYS_SETUP.md",
    "CONTRIBUTING.md",
    "TESTING_GUIDE.md",
    "LICENSE",
    "CODEBASE_ANALYSIS.md",
]

# Files to move to docs/
MOVE_TO_DOCS = [
    ("QUICK_START.md", "docs/QUICKSTART.md"),
    ("API_KEYS_SETUP.md", "docs/API_KEYS_SETUP.md"),
    ("TESTING_GUIDE.md", "docs/TESTING_GUIDE.md"),
]


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def cleanup(execute: bool = False, verbose: bool = True):
    """Perform cleanup operations."""
    root = get_project_root()
    
    print("=" * 60)
    print("SciTrans-LLMs NEW - Cleanup Script")
    print("=" * 60)
    print()
    
    if not execute:
        print("[DRY RUN] No files will be deleted. Use --execute to actually delete.")
        print()
    
    # Count operations
    deleted = 0
    moved = 0
    skipped = 0
    
    # Delete redundant files
    print("Files to DELETE:")
    print("-" * 40)
    for filename in REDUNDANT_FILES:
        filepath = root / filename
        if filepath.exists():
            if execute:
                filepath.unlink()
                print(f"  [DELETED] {filename}")
            else:
                print(f"  [WOULD DELETE] {filename}")
            deleted += 1
        else:
            if verbose:
                print(f"  [SKIP] {filename} (not found)")
            skipped += 1
    
    print()
    
    # Move files to docs/
    print("Files to MOVE:")
    print("-" * 40)
    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    for src, dst in MOVE_TO_DOCS:
        src_path = root / src
        dst_path = root / dst
        
        if src_path.exists() and not dst_path.exists():
            if execute:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                src_path.rename(dst_path)
                print(f"  [MOVED] {src} -> {dst}")
            else:
                print(f"  [WOULD MOVE] {src} -> {dst}")
            moved += 1
        elif dst_path.exists():
            if verbose:
                print(f"  [SKIP] {src} (destination exists)")
            skipped += 1
        else:
            if verbose:
                print(f"  [SKIP] {src} (not found)")
            skipped += 1
    
    print()
    
    # Clean up __pycache__ directories
    print("Cache directories to CLEAN:")
    print("-" * 40)
    cache_count = 0
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            if execute:
                import shutil
                shutil.rmtree(pycache)
                print(f"  [DELETED] {pycache.relative_to(root)}")
            else:
                print(f"  [WOULD DELETE] {pycache.relative_to(root)}")
            cache_count += 1
    
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Files to delete: {deleted}")
    print(f"  Files to move: {moved}")
    print(f"  Cache dirs to clean: {cache_count}")
    print(f"  Skipped: {skipped}")
    print()
    
    if not execute:
        print("Run with --execute to perform these operations.")
    else:
        print("Cleanup complete!")
    
    return deleted + moved + cache_count


def main():
    parser = argparse.ArgumentParser(description="Cleanup SciTrans-LLMs project")
    parser.add_argument("--execute", "-x", action="store_true", 
                       help="Actually delete files (default is dry run)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    args = parser.parse_args()
    
    cleanup(execute=args.execute, verbose=not args.quiet)


if __name__ == "__main__":
    main()


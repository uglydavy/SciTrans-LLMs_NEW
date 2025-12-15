#!/usr/bin/env python3
"""
Identify deprecated and unused files in the codebase.

Usage:
    python scripts/identify_deprecated.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Set
try:
    import ast
except ImportError:
    ast = None

# Files/folders to check for deprecation
DEPRECATED_FILES = [
    "scitran/utils/batch_translator.py",  # Superseded by fast_translator.py
    "QUICK_START.md",
    "QUICK_START_GUI.md",
    "QUICK_FIX_TRANSLATION.md",
    "OLLAMA_FIX.md",
    "setup.sh",
    "QUICK_INSTALL.sh",
    "quickstart.py",
    "run.py",  # If exists, superseded by scitrans CLI
]

# Patterns to search for in code
DEPRECATED_PATTERNS = [
    r"from scitran\.utils\.batch_translator import",
    r"import batch_translator",
    r"TODO|FIXME|XXX|HACK",
    r"deprecated|DEPRECATED",
]


def find_file_references(file_path: Path, search_patterns: List[str]) -> Dict[str, List[str]]:
    """Find references to deprecated patterns in a file."""
    references = {}
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        for pattern in search_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                references[pattern] = matches
    except Exception as e:
        pass
    
    return references


def find_unused_imports(file_path: Path) -> List[str]:
    """Find unused imports in a Python file."""
    unused = []
    
    if ast is None:
        return unused
    
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content, filename=str(file_path))
        
        # Get all imports
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # Get all names used
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        
        # Find unused imports (simplified check)
        for imp in imports:
            if imp not in names and imp not in ['sys', 'os', 'pathlib', 'typing']:
                # More sophisticated check needed, but this is a start
                pass
    except Exception:
        pass
    
    return unused


def check_deprecated_files() -> Dict[str, bool]:
    """Check which deprecated files exist."""
    results = {}
    project_root = Path(__file__).parent.parent
    
    for file_path in DEPRECATED_FILES:
        full_path = project_root / file_path
        results[file_path] = full_path.exists()
    
    return results


def find_commented_code() -> List[Dict]:
    """Find large blocks of commented code."""
    commented_blocks = []
    project_root = Path(__file__).parent.parent
    
    for py_file in project_root.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            in_comment_block = False
            comment_start = 0
            comment_lines = []
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Check for large comment blocks
                if stripped.startswith('#') and len(stripped) > 5:
                    if not in_comment_block:
                        in_comment_block = True
                        comment_start = i
                        comment_lines = [line]
                    else:
                        comment_lines.append(line)
                else:
                    if in_comment_block and len(comment_lines) > 10:
                        commented_blocks.append({
                            'file': str(py_file.relative_to(project_root)),
                            'start_line': comment_start + 1,
                            'end_line': i,
                            'lines': len(comment_lines)
                        })
                    in_comment_block = False
                    comment_lines = []
        except Exception:
            pass
    
    return commented_blocks


def main():
    """Main analysis."""
    print("="*60)
    print("DEPRECATED CODE ANALYSIS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    
    # Check deprecated files
    print("\n1. DEPRECATED FILES")
    print("-" * 60)
    deprecated_files = check_deprecated_files()
    
    for file_path, exists in deprecated_files.items():
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  {file_path:50} {status}")
    
    # Find references to deprecated code
    print("\n2. REFERENCES TO DEPRECATED CODE")
    print("-" * 60)
    
    references_found = {}
    for py_file in project_root.rglob("*.py"):
        refs = find_file_references(py_file, DEPRECATED_PATTERNS)
        if refs:
            references_found[str(py_file.relative_to(project_root))] = refs
    
    if references_found:
        for file_path, refs in references_found.items():
            print(f"\n  {file_path}:")
            for pattern, matches in refs.items():
                print(f"    - {pattern}: {len(matches)} matches")
    else:
        print("  No references to deprecated patterns found")
    
    # Find commented code blocks
    print("\n3. LARGE COMMENTED CODE BLOCKS")
    print("-" * 60)
    
    commented = find_commented_code()
    if commented:
        for block in commented[:10]:  # Show first 10
            print(f"  {block['file']}:{block['start_line']}-{block['end_line']} ({block['lines']} lines)")
    else:
        print("  No large commented blocks found")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    deprecated_count = sum(1 for v in deprecated_files.values() if v)
    print(f"Deprecated files found: {deprecated_count}/{len(DEPRECATED_FILES)}")
    print(f"Files with deprecated references: {len(references_found)}")
    print(f"Large commented blocks: {len(commented)}")
    
    print("\nRECOMMENDATIONS:")
    print("1. Archive or remove deprecated files")
    print("2. Update code that references deprecated modules")
    print("3. Remove or document large commented blocks")
    print("4. Run 'ruff --select=F401,F841 .' to find unused imports")


if __name__ == "__main__":
    main()

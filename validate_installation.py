#!/usr/bin/env python3
"""
Installation validation script for SciTrans-LLMs.
Run this to verify all components are properly installed.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check if all critical imports work."""
    print("=" * 70)
    print("CHECKING IMPORTS")
    print("=" * 70)
    
    checks = []
    
    # Core
    try:
        from scitran.core.models import Document, Block
        print("✓ Core models")
        checks.append(True)
    except Exception as e:
        print(f"✗ Core models: {e}")
        checks.append(False)
    
    try:
        from scitran.core.pipeline import TranslationPipeline, PipelineConfig
        print("✓ Pipeline")
        checks.append(True)
    except Exception as e:
        print(f"✗ Pipeline: {e}")
        checks.append(False)
    
    # Masking
    try:
        from scitran.masking.engine import MaskingEngine
        print("✓ Masking engine")
        checks.append(True)
    except Exception as e:
        print(f"✗ Masking engine: {e}")
        checks.append(False)
    
    # Translation
    try:
        from scitran.translation.prompts import PromptOptimizer
        from scitran.translation.base import TranslationBackend
        print("✓ Translation system")
        checks.append(True)
    except Exception as e:
        print(f"✗ Translation system: {e}")
        checks.append(False)
    
    # Backends
    try:
        from scitran.translation.backends import (
            OpenAIBackend, AnthropicBackend, DeepSeekBackend,
            OllamaBackend, FreeBackend
        )
        print("✓ Translation backends")
        checks.append(True)
    except Exception as e:
        print(f"✗ Translation backends: {e}")
        checks.append(False)
    
    # PDF Processing
    try:
        from scitran.extraction.pdf_parser import PDFParser
        from scitran.rendering.pdf_renderer import PDFRenderer
        print("✓ PDF processing")
        checks.append(True)
    except Exception as e:
        print(f"✗ PDF processing: {e}")
        checks.append(False)
    
    # Scoring
    try:
        from scitran.scoring.reranker import AdvancedReranker
        print("✓ Reranking system")
        checks.append(True)
    except Exception as e:
        print(f"✗ Reranking system: {e}")
        checks.append(False)
    
    # Utils
    try:
        from scitran.utils import setup_logger, TranslationCache, load_config
        print("✓ Utilities")
        checks.append(True)
    except Exception as e:
        print(f"✗ Utilities: {e}")
        checks.append(False)
    
    return all(checks)


def check_backends():
    """Check which backends are available."""
    print("\n" + "=" * 70)
    print("CHECKING TRANSLATION BACKENDS")
    print("=" * 70)
    
    from scitran.translation.backends import (
        OpenAIBackend, AnthropicBackend, DeepSeekBackend,
        OllamaBackend, FreeBackend
    )
    
    backends = [
        ("OpenAI", OpenAIBackend, "gpt-4o"),
        ("Anthropic", AnthropicBackend, "claude-3-5-sonnet-20241022"),
        ("DeepSeek", DeepSeekBackend, "deepseek-chat"),
        ("Ollama", OllamaBackend, "llama3.1"),
        ("Free (Google)", FreeBackend, "google")
    ]
    
    available_count = 0
    
    for name, backend_class, model in backends:
        try:
            backend = backend_class(model=model)
            if backend.is_available():
                print(f"✓ {name:20} - READY")
                available_count += 1
            else:
                print(f"⚠ {name:20} - NOT CONFIGURED (API key missing)")
        except Exception as e:
            print(f"✗ {name:20} - ERROR: {str(e)[:40]}")
    
    print(f"\nAvailable backends: {available_count}/5")
    
    if available_count == 0:
        print("\n⚠️  WARNING: No backends configured!")
        print("   Set API keys:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
    
    return available_count > 0


def check_dependencies():
    """Check optional dependencies."""
    print("\n" + "=" * 70)
    print("CHECKING OPTIONAL DEPENDENCIES")
    print("=" * 70)
    
    deps = [
        ("PyMuPDF (fitz)", "fitz"),
        ("pdfplumber", "pdfplumber"),
        ("OpenAI", "openai"),
        ("Anthropic", "anthropic"),
        ("Ollama", "ollama"),
        ("Deep Translator", "deep_translator"),
        ("Gradio", "gradio"),
        ("Loguru", "loguru"),
        ("DiskCache", "diskcache"),
    ]
    
    installed = 0
    
    for name, module in deps:
        try:
            __import__(module)
            print(f"✓ {name}")
            installed += 1
        except ImportError:
            print(f"✗ {name} - Not installed")
    
    print(f"\nInstalled: {installed}/{len(deps)}")
    
    if installed < len(deps):
        print("\n💡 To install missing packages:")
        print("   pip install -r requirements-minimal.txt")
        print("   pip install -r requirements-ml.txt")


def check_files():
    """Check if critical files exist."""
    print("\n" + "=" * 70)
    print("CHECKING PROJECT FILES")
    print("=" * 70)
    
    critical_files = [
        "scitran/__init__.py",
        "scitran/core/models.py",
        "scitran/core/pipeline.py",
        "scitran/translation/base.py",
        "scitran/translation/backends/__init__.py",
        "scitran/masking/engine.py",
        "configs/default.yaml",
        "cli/commands/main.py",
        "README.md",
        "Makefile",
        "pyproject.toml"
    ]
    
    all_exist = True
    
    for file_path in critical_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def run_quick_test():
    """Run a quick functional test."""
    print("\n" + "=" * 70)
    print("RUNNING QUICK FUNCTIONAL TEST")
    print("=" * 70)
    
    try:
        from scitran.core.models import Block, Document
        from scitran.masking.engine import MaskingEngine
        
        # Test masking
        block = Block(
            block_id="test",
            source_text="The equation $E = mc^2$ is famous."
        )
        
        engine = MaskingEngine()
        masked = engine.mask_block(block)
        
        if len(masked.masks) > 0 and "MASK_" in masked.masked_text:
            print("✓ Masking test passed")
            
            # Test unmasking
            masked.translated_text = masked.masked_text.replace("equation", "équation")
            unmasked = engine.unmask_block(masked)
            
            if "$E = mc^2$" in unmasked.translated_text:
                print("✓ Unmasking test passed")
                return True
        
        print("✗ Functional test failed")
        return False
        
    except Exception as e:
        print(f"✗ Functional test error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SciTrans-LLMs Installation Validator" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    checks = {
        "Imports": check_imports(),
        "Backends": check_backends(),
        "Files": check_files(),
        "Functional Test": run_quick_test()
    }
    
    check_dependencies()  # Info only, doesn't affect pass/fail
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Installation is valid!")
        print("\nNext steps:")
        print("  1. Set API keys: export OPENAI_API_KEY='sk-...'")
        print("  2. Run GUI: python gui/app.py")
        print("  3. Try CLI: python -m cli.commands.main --help")
        print("  4. Read docs: cat docs/QUICKSTART.md")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements-minimal.txt")
        print("  2. Check Python version: python --version (need 3.9+)")
        print("  3. Run setup script: ./setup.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())

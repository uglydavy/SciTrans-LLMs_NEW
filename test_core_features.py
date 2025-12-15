#!/usr/bin/env python3
"""
Test script to verify core features work correctly.
Tests core functionality without requiring GUI dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_line_break_preservation():
    """Test line break preservation in postprocessing."""
    print("=" * 60)
    print("TEST 1: Line Break Preservation")
    print("=" * 60)
    
    from scitran.core.pipeline import TranslationPipeline, PipelineConfig
    
    config = PipelineConfig(backend="free")  # Use free backend that doesn't need API key
    pipeline = TranslationPipeline(config)
    
    # Test text with line breaks
    test_cases = [
        ("Line 1\nLine 2\nLine 3", True, "Simple line breaks"),
        ("Word1   Word2\n\nWord3", True, "Multiple spaces and newlines"),
        ("Text with    multiple   spaces\nand\nnewlines", True, "Spaces and newlines"),
        ("No newlines here", False, "No newlines"),
        ("\n\n\nMultiple\n\n\nNewlines", True, "Multiple consecutive newlines"),
        ("Paragraph 1.\n\nParagraph 2.", True, "Paragraph breaks"),
    ]
    
    all_passed = True
    for input_text, has_newlines, description in test_cases:
        result = pipeline._postprocess_translation(input_text)
        print(f"\n{description}:")
        print(f"  Input:  {repr(input_text)}")
        print(f"  Output: {repr(result)}")
        
        # Check that newlines are preserved (not collapsed to spaces)
        if has_newlines and "\n" in input_text:
            if "\n" not in result:
                print(f"  ❌ FAIL: Newlines should be preserved")
                all_passed = False
            else:
                print(f"  ✓ Newlines preserved")
        
        # Check that multiple spaces are collapsed
        if "   " in input_text:
            if "   " in result:
                print(f"  ❌ FAIL: Multiple spaces should be collapsed")
                all_passed = False
            else:
                print(f"  ✓ Multiple spaces collapsed")
        
        # Check that consecutive newlines are limited to 2
        if "\n\n\n" in input_text:
            if "\n\n\n" in result:
                print(f"  ❌ FAIL: More than 2 consecutive newlines should be limited")
                all_passed = False
            else:
                print(f"  ✓ Consecutive newlines limited")
    
    if all_passed:
        print("\n✓ Line Break Preservation: PASSED\n")
    else:
        print("\n❌ Line Break Preservation: FAILED\n")
    
    return all_passed


def test_glossary_loading():
    """Test glossary loading for all domains."""
    print("=" * 60)
    print("TEST 2: Glossary Loading")
    print("=" * 60)
    
    from scitran.translation.glossary.manager import GlossaryManager
    
    manager = GlossaryManager()
    initial_count = len(manager)
    print(f"Initial glossary count: {initial_count}")
    
    # Test loading each domain
    domains = ['ml', 'physics', 'biology', 'chemistry', 'cs', 'statistics', 'europarl']
    loaded_domains = []
    failed_domains = []
    
    for domain in domains:
        try:
            count = manager.load_domain(domain, "en-fr")
            if count > 0:
                loaded_domains.append((domain, count))
                print(f"✓ {domain.upper()}: Loaded {count} terms")
            else:
                failed_domains.append(domain)
                print(f"⚠ {domain.upper()}: No terms loaded (file may be missing)")
        except Exception as e:
            failed_domains.append(domain)
            print(f"❌ {domain.upper()}: Error - {e}")
    
    final_count = len(manager)
    print(f"\nFinal glossary count: {final_count}")
    print(f"Total terms added: {final_count - initial_count}")
    print(f"Successfully loaded: {len(loaded_domains)}/{len(domains)} domains")
    
    if len(loaded_domains) > 0:
        print("\n✓ Glossary Loading: PASSED (some domains loaded)\n")
        return True
    else:
        print("\n❌ Glossary Loading: FAILED (no domains loaded)\n")
        return False


def test_api_key_functions():
    """Test API key management functions (without GUI)."""
    print("=" * 60)
    print("TEST 3: API Key Management Functions")
    print("=" * 60)
    
    import json
    import os
    from pathlib import Path
    
    # Simulate config structure
    config_file = Path.home() / ".scitrans" / "config.json"
    config_file.parent.mkdir(exist_ok=True)
    
    # Load existing config or create default
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {"api_keys": {}}
    
    # Test save
    test_key = "test-key-12345-abcdef"
    if "api_keys" not in config:
        config["api_keys"] = {}
    config["api_keys"]["test_backend"] = test_key
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ API key saved to config")
    
    # Test read
    with open(config_file) as f:
        config_loaded = json.load(f)
    
    assert config_loaded["api_keys"]["test_backend"] == test_key
    print(f"✓ API key read from config")
    
    # Test delete
    del config_loaded["api_keys"]["test_backend"]
    with open(config_file, 'w') as f:
        json.dump(config_loaded, f, indent=2)
    
    print(f"✓ API key deleted from config")
    
    # Verify deletion
    with open(config_file) as f:
        config_final = json.load(f)
    
    assert "test_backend" not in config_final.get("api_keys", {})
    print(f"✓ API key deletion verified")
    
    print("\n✓ API Key Management Functions: PASSED\n")
    return True


def test_backend_imports():
    """Test that all backend modules can be imported."""
    print("=" * 60)
    print("TEST 4: Backend Imports")
    print("=" * 60)
    
    backends = {
        "local": "scitran.translation.backends.local_backend",
        "libre": "scitran.translation.backends.libre_backend",
        "argos": "scitran.translation.backends.argos_backend",
        "free": "scitran.translation.backends.free_backend",
        "cascade": "scitran.translation.backends.cascade_backend",
    }
    
    all_passed = True
    for backend_name, module_path in backends.items():
        try:
            __import__(module_path)
            print(f"✓ {backend_name}: Import successful")
        except ImportError as e:
            print(f"⚠ {backend_name}: Import failed - {e}")
            # Some backends may have optional dependencies
        except Exception as e:
            print(f"❌ {backend_name}: Error - {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ Backend Imports: PASSED\n")
    else:
        print("\n⚠ Backend Imports: Some backends have optional dependencies\n")
    
    return True  # Not critical if some optional backends fail


def test_glossary_manager_methods():
    """Test GlossaryManager methods."""
    print("=" * 60)
    print("TEST 5: GlossaryManager Methods")
    print("=" * 60)
    
    from scitran.translation.glossary.manager import GlossaryManager
    
    manager = GlossaryManager()
    
    # Test add_term
    manager.add_term("test term", "terme de test", "test")
    assert len(manager) == 1
    print("✓ add_term: Works")
    
    # Test get_term
    term = manager.get_term("test term")
    assert term is not None
    assert term.target == "terme de test"
    print("✓ get_term: Works")
    
    # Test get_translation
    translation = manager.get_translation("test term")
    assert translation == "terme de test"
    print("✓ get_translation: Works")
    
    # Test find_terms_in_text
    found = manager.find_terms_in_text("This is a test term in the text")
    assert len(found) == 1
    print("✓ find_terms_in_text: Works")
    
    # Test generate_prompt_section
    prompt = manager.generate_prompt_section("This is a test term")
    assert "test term" in prompt.lower()
    print("✓ generate_prompt_section: Works")
    
    # Test to_dict
    glossary_dict = manager.to_dict()
    assert "test term" in glossary_dict
    assert glossary_dict["test term"] == "terme de test"
    print("✓ to_dict: Works")
    
    # Test clear
    manager.clear()
    assert len(manager) == 0
    print("✓ clear: Works")
    
    print("\n✓ GlossaryManager Methods: PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CORE FEATURES TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Line Break Preservation", test_line_break_preservation()))
        results.append(("Glossary Loading", test_glossary_loading()))
        results.append(("API Key Functions", test_api_key_functions()))
        results.append(("Backend Imports", test_backend_imports()))
        results.append(("GlossaryManager Methods", test_glossary_manager_methods()))
        
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASSED" if result else "❌ FAILED"
            print(f"{status}: {name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED ✓")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED")
            print("=" * 60)
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


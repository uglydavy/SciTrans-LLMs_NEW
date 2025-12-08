#!/usr/bin/env python3
"""
Quick start script for SciTrans-LLMs NEW.
Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from scitran.core.models import Document, Block, Segment
        print("✅ Core models")
    except ImportError as e:
        print(f"❌ Core models: {e}")
        return False
    
    try:
        from scitran.masking.engine import MaskingEngine
        print("✅ Masking engine")
    except ImportError as e:
        print(f"❌ Masking engine: {e}")
        return False
    
    try:
        from scitran.translation.prompts import PromptOptimizer
        print("✅ Prompt system")
    except ImportError as e:
        print(f"❌ Prompt system: {e}")
        return False
    
    try:
        from scitran.scoring.reranker import AdvancedReranker
        print("✅ Reranking system")
    except ImportError as e:
        print(f"❌ Reranking system: {e}")
        return False
    
    try:
        from scitran.core.pipeline import TranslationPipeline
        print("✅ Translation pipeline")
    except ImportError as e:
        print(f"❌ Translation pipeline: {e}")
        return False
    
    return True


def test_masking():
    """Test the masking engine."""
    print("\nTesting masking engine...")
    
    from scitran.core.models import Block
    from scitran.masking.engine import MaskingEngine
    
    # Create test block
    block = Block(
        block_id="test_1",
        source_text="The equation $E = mc^2$ shows that energy equals mass times the speed of light squared. See https://example.com for details."
    )
    
    # Apply masking
    engine = MaskingEngine()
    masked_block = engine.mask_block(block)
    
    print(f"Original: {block.source_text}")
    print(f"Masked: {masked_block.masked_text}")
    print(f"Masks applied: {len(masked_block.masks)}")
    
    # Test unmasking
    masked_block.translated_text = masked_block.masked_text.replace("equation", "équation")
    unmasked_block = engine.unmask_block(masked_block)
    
    print(f"Unmasked: {unmasked_block.translated_text}")
    
    return len(masked_block.masks) > 0


def test_prompt_generation():
    """Test prompt generation."""
    print("\nTesting prompt generation...")
    
    from scitran.core.models import Block
    from scitran.translation.prompts import PromptOptimizer
    
    block = Block(
        block_id="test_2",
        source_text="Machine learning has revolutionized natural language processing."
    )
    
    optimizer = PromptOptimizer()
    system_prompt, user_prompt = optimizer.generate_prompt(
        block=block,
        template_name="scientific_expert",
        source_lang="en",
        target_lang="fr",
        glossary_terms={"machine learning": "apprentissage automatique"},
        context=[]
    )
    
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"User prompt length: {len(user_prompt)} chars")
    print(f"Best template: {optimizer.select_best_template()}")
    
    return len(system_prompt) > 0 and len(user_prompt) > 0


def test_reranking():
    """Test the reranking system."""
    print("\nTesting reranking system...")
    
    from scitran.core.models import Block
    from scitran.scoring.reranker import AdvancedReranker
    
    block = Block(
        block_id="test_3",
        source_text="Deep learning enables state-of-the-art performance."
    )
    
    candidates = [
        "L'apprentissage profond permet des performances de pointe.",
        "Deep learning permet performance état de l'art.",
        "L'apprentissage profond active la performance d'état de l'art."
    ]
    
    reranker = AdvancedReranker()
    best, scored = reranker.rerank(
        candidates=candidates,
        block=block,
        glossary={"deep learning": "apprentissage profond"},
        context=[]
    )
    
    print(f"Best translation: {best}")
    print(f"Scores: {[f'{c.total_score:.2f}' for c in scored]}")
    
    return best == candidates[0]  # First should be best


def test_pipeline_config():
    """Test pipeline configuration."""
    print("\nTesting pipeline configuration...")
    
    from scitran.core.pipeline import PipelineConfig
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai",
        num_candidates=3,
        enable_masking=True,
        enable_context=True,
        preserve_layout=True
    )
    
    issues = config.validate()
    
    if issues:
        print(f"❌ Configuration issues: {issues}")
        return False
    else:
        print("✅ Configuration valid")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SciTrans-LLMs NEW - Quick Start Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        all_passed = False
    
    # Test masking
    try:
        if not test_masking():
            print("\n❌ Masking test failed.")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Masking test error: {e}")
        all_passed = False
    
    # Test prompts
    try:
        if not test_prompt_generation():
            print("\n❌ Prompt generation test failed.")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Prompt generation error: {e}")
        all_passed = False
    
    # Test reranking
    try:
        if not test_reranking():
            print("\n❌ Reranking test failed.")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Reranking error: {e}")
        all_passed = False
    
    # Test pipeline config
    try:
        if not test_pipeline_config():
            print("\n❌ Pipeline config test failed.")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Pipeline config error: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All tests passed! The system is ready.")
        print("\nNext steps:")
        print("1. Set your API keys:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\n2. Launch the GUI:")
        print("   python gui/app.py")
        print("\n3. Or use the API:")
        print("   from scitran.core.pipeline import TranslationPipeline")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTry running:")
        print("   ./setup.sh")
        print("to ensure all dependencies are installed.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

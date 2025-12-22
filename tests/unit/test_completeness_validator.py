# -*- coding: utf-8 -*-
"""
Tests for TranslationCompletenessValidator.

Tests:
- Coverage validation
- Identity translation detection
- Mask preservation validation
- Strict validation mode
"""

import pytest
from scitran.core.validator import TranslationCompletenessValidator, ValidationResult
from scitran.core.models import Document, Segment, Block, BlockType, MaskInfo


def create_test_document(blocks: list) -> Document:
    """Create a test document with given blocks."""
    segment = Segment(
        segment_id="test_segment",
        segment_type="section",
        blocks=blocks
    )
    return Document(
        document_id="test_doc",
        segments=[segment]
    )


def test_full_coverage_valid():
    """Test that full coverage passes validation."""
    blocks = [
        Block(
            block_id="block_1",
            source_text="Hello world",
            translated_text="Bonjour le monde",
            block_type=BlockType.PARAGRAPH
        ),
        Block(
            block_id="block_2",
            source_text="Machine learning",
            translated_text="Apprentissage automatique",
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator()
    
    result = validator.validate(document)
    
    assert result.is_valid
    assert result.coverage == 1.0
    assert len(result.errors) == 0
    assert len(result.failed_blocks) == 0


def test_missing_translation_invalid():
    """Test that missing translations fail validation."""
    blocks = [
        Block(
            block_id="block_1",
            source_text="Hello world",
            translated_text="Bonjour le monde",
            block_type=BlockType.PARAGRAPH
        ),
        Block(
            block_id="block_2",
            source_text="Missing translation",
            translated_text=None,  # Missing!
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator()
    
    result = validator.validate(document)
    
    assert not result.is_valid
    assert result.coverage == 0.5
    assert len(result.failed_blocks) == 1
    assert "block_2" in result.failed_blocks
    assert any("Missing translation" in err for err in result.errors)


def test_identity_translation_invalid():
    """Test that validator catches REAL identity translation failures (smart heuristic)."""
    # Use long text to trigger smart heuristic (>30 chars, >6 words)
    blocks = [
        Block(
            block_id="block_1",
            source_text="This is a long paragraph with more than thirty characters and many words that clearly should be translated.",
            translated_text="This is a long paragraph with more than thirty characters and many words that clearly should be translated.",  # Identity!
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator(
        require_no_identity=True,
        source_lang="en",
        target_lang="fr"
    )
    
    result = validator.validate(document)
    
    assert not result.is_valid
    assert len(result.identity_blocks) == 1
    assert "block_1" in result.identity_blocks


def test_missing_mask_invalid():
    """Test that missing masks fail validation (POST-unmask check)."""
    from scitran.core.models import TranslationMetadata
    from datetime import datetime
    
    mask = MaskInfo(
        original="E = mc^2",
        placeholder="<<MATH_0001>>",
        mask_type="MATH"
    )
    
    # Simulate POST-unmask state where restoration failed
    metadata = TranslationMetadata(
        backend="test",
        timestamp=datetime.now(),
        duration=0.0,
        restored_masks=0,  # Failed to restore
        missing_placeholders=["<<MATH_0001>>"]  # This placeholder was missing
    )
    
    blocks = [
        Block(
            block_id="block_1",
            source_text="The equation E = mc^2 is important",  # After unmask attempt
            translated_text="L'équation est importante",  # Mask was missing, unmask did nothing
            masks=[mask],
            metadata=metadata,  # Restoration metadata shows failure
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator(require_mask_preservation=True)
    
    result = validator.validate(document)
    
    assert not result.is_valid
    assert len(result.missing_masks) == 1
    assert result.missing_masks[0]["placeholder"] == "<<MATH_0001>>"
    assert any("restore" in err.lower() for err in result.errors)


def test_mask_preserved_valid():
    """Test that preserved masks pass validation."""
    mask = MaskInfo(
        original="E = mc^2",
        placeholder="<<MATH_0001>>",
        mask_type="MATH"
    )
    
    blocks = [
        Block(
            block_id="block_1",
            source_text="The equation <<MATH_0001>> is important",
            translated_text="L'équation <<MATH_0001>> est importante",  # Mask preserved!
            masks=[mask],
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator(require_mask_preservation=True)
    
    result = validator.validate(document)
    
    assert result.is_valid
    assert result.total_masks == 1
    assert result.preserved_masks == 1
    assert len(result.missing_masks) == 0


def test_non_translatable_blocks_ignored():
    """Test that non-translatable blocks are ignored in coverage."""
    blocks = [
        Block(
            block_id="block_1",
            source_text="Hello",
            translated_text="Bonjour",
            block_type=BlockType.PARAGRAPH  # Translatable
        ),
        Block(
            block_id="block_2",
            source_text="E = mc^2",
            translated_text=None,  # No translation (equation - not translatable)
            block_type=BlockType.EQUATION  # Not translatable
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator()
    
    result = validator.validate(document)
    
    # Should only count translatable blocks
    assert result.translatable_blocks == 1
    assert result.translated_blocks == 1
    assert result.coverage == 1.0
    assert result.is_valid


def test_validation_report_generation():
    """Test that validation report is generated."""
    blocks = [
        Block(
            block_id="block_1",
            source_text="Hello",
            translated_text=None,
            block_type=BlockType.PARAGRAPH
        ),
    ]
    
    document = create_test_document(blocks)
    validator = TranslationCompletenessValidator()
    
    result = validator.validate(document)
    report = validator.generate_report(result)
    
    assert "VALIDATION REPORT" in report
    assert "INVALID" in report or "✗" in report
    assert "Coverage" in report


def test_allow_partial_coverage():
    """Test that partial coverage is allowed when configured."""
    blocks = [
        Block(block_id="block_1", source_text="Hello", translated_text="Bonjour", block_type=BlockType.PARAGRAPH),
        Block(block_id="block_2", source_text="World", translated_text=None, block_type=BlockType.PARAGRAPH),
    ]
    
    document = create_test_document(blocks)
    
    # Strict validator: should fail
    strict_validator = TranslationCompletenessValidator(require_full_coverage=True)
    strict_result = strict_validator.validate(document)
    assert not strict_result.is_valid
    
    # Lenient validator: should pass
    lenient_validator = TranslationCompletenessValidator(require_full_coverage=False)
    lenient_result = lenient_validator.validate(document)
    # Even though coverage < 1.0, lenient mode doesn't set is_valid=False for coverage alone
    assert lenient_result.coverage == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


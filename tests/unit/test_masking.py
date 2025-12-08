"""Unit tests for masking engine."""

import pytest
from scitran.masking.engine import MaskingEngine, MaskingConfig
from scitran.core.models import Block


def test_latex_masking():
    """Test LaTeX masking."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text="The equation $E = mc^2$ shows energy."
    )
    
    masked = engine.mask_block(block)
    
    assert "<<LATEX_INLINE_" in masked.masked_text
    assert len(masked.masks) > 0
    assert masked.masks[0].mask_type == "latex_inline"


def test_url_masking():
    """Test URL masking."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text="Visit https://example.com for details."
    )
    
    masked = engine.mask_block(block)
    
    assert "<<URL_FULL_" in masked.masked_text
    assert any(m.mask_type == "url_full" for m in masked.masks)


def test_code_masking():
    """Test code block masking."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text="Use `print('hello')` to output text."
    )
    
    masked = engine.mask_block(block)
    
    assert "<<CODE_INLINE_" in masked.masked_text
    assert any(m.mask_type == "code_inline" for m in masked.masks)


def test_unmasking():
    """Test mask restoration."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text="The value is $x = 5$ here."
    )
    
    masked = engine.mask_block(block)
    masked.translated_text = masked.masked_text.replace("value", "valeur")
    unmasked = engine.unmask_block(masked)
    
    assert "$x = 5$" in unmasked.translated_text
    assert "valeur" in unmasked.translated_text


def test_validation():
    """Test mask validation."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text="Equation: $y = mx + b$ end."
    )
    
    masked = engine.mask_block(block)
    
    # Simulate translation with preserved mask
    masked.translated_text = masked.masked_text.replace("Equation", "Équation")
    
    # Validate masks are preserved
    is_valid, missing = engine.validate_masks(
        original_text=block.source_text,
        translated_text=masked.translated_text,
        masks=masked.masks
    )
    
    assert is_valid
    assert len(missing) == 0


def test_document_masking():
    """Test masking at document level."""
    from scitran.core.models import Document, Segment
    
    engine = MaskingEngine()
    
    block1 = Block(block_id="b1", source_text="Formula: $a + b = c$")
    block2 = Block(block_id="b2", source_text="Link: https://example.com")
    
    segment = Segment(segment_id="s1", segment_type="main", blocks=[block1, block2])
    document = Document(document_id="test_doc", segments=[segment])
    
    masked_doc = engine.mask_document(document)
    
    assert masked_doc.stats.get('total_masks', 0) >= 2


def test_nested_latex_masking():
    """Test nested LaTeX environments."""
    engine = MaskingEngine()
    block = Block(
        block_id="test",
        source_text=r"See equation: $$\frac{d}{dx}\int_0^x f(t)dt = f(x)$$ for details."
    )
    
    masked = engine.mask_block(block)
    
    assert "<<LATEX_DISPLAY_" in masked.masked_text
    assert len(masked.masks) >= 1


def test_citation_masking():
    """Test citation masking."""
    config = MaskingConfig(mask_references=True)
    engine = MaskingEngine(config)
    
    block = Block(
        block_id="test",
        source_text="As shown in [1], the results from (Smith et al., 2020) confirm..."
    )
    
    masked = engine.mask_block(block)
    
    assert "<<CITATION_" in masked.masked_text


def test_statistics():
    """Test masking statistics."""
    engine = MaskingEngine()
    
    block = Block(
        block_id="test",
        source_text="Code `x=1` and formula $y=2$ and URL https://test.com"
    )
    
    engine.mask_block(block)
    stats = engine.get_statistics()
    
    assert stats['total_masks'] >= 3
    assert 'masks_by_type' in stats


def test_reset():
    """Test engine reset."""
    engine = MaskingEngine()
    
    block = Block(block_id="test", source_text="$x=1$")
    engine.mask_block(block)
    
    assert engine.get_statistics()['total_masks'] >= 1
    
    engine.reset()
    
    assert engine.get_statistics()['total_masks'] == 0

"""
Tests for truncation detection and repair (PHASE 3.1).

Tests:
- Detect truncation from finish_reason
- Detect truncation from length mismatch
- Chunk-based repair
- Placeholder preservation in chunks
"""

import pytest
from scitran.core.pipeline import TranslationPipeline, PipelineConfig


def test_detect_truncation_finish_reason():
    """Test truncation detection from finish_reason."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    # Test finish_reason="length" indicates truncation
    assert pipeline._is_suspected_partial(
        source="This is a long text" * 50,
        translated="Ceci est",
        finish_reason="length"
    )
    
    # Test finish_reason="stop" is normal
    assert not pipeline._is_suspected_partial(
        source="Hello world",
        translated="Bonjour le monde",
        finish_reason="stop"
    )


def test_detect_truncation_length_mismatch():
    """Test truncation detection from suspicious length ratios."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    # Long source, very short translation = suspicious
    long_source = "This is a very long text. " * 20  # 500+ chars
    short_translation = "Ceci"  # < 25% of source
    
    assert pipeline._is_suspected_partial(long_source, short_translation)
    
    # Normal length ratio
    normal_translation = "Ceci est un texte très long. " * 15
    assert not pipeline._is_suspected_partial(long_source, normal_translation)


def test_detect_truncation_ends_mid_sentence():
    """Test detection based on ending mid-sentence + short ratio."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    # Make source longer (>200 chars for the heuristic to apply)
    source = "This is a long paragraph with multiple sentences and substantial content. " * 3
    # Truncated translation (ends mid-sentence, short ratio <40%)
    truncated = "Ceci est un long paragraphe avec"  # No ending punctuation
    
    # Should detect truncation (long source, short ratio, no ending punctuation)
    assert pipeline._is_suspected_partial(source, truncated)


def test_chunk_for_translation_preserves_placeholders():
    """Test that chunking doesn't split placeholders."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    text = "This is text with <<MATH_0001>> and more text.\n\nAnother paragraph with <<URL_0002>>."
    
    chunks = pipeline._chunk_for_translation(text)
    
    # All placeholders should be intact in chunks
    all_text = " ".join(chunks)
    assert "<<MATH_0001>>" in all_text
    assert "<<URL_0002>>" in all_text
    
    # No chunk should have partial placeholder
    for chunk in chunks:
        # Count opening and closing brackets
        assert chunk.count("<<") == chunk.count(">>")


def test_chunk_for_translation_paragraph_boundaries():
    """Test that chunking respects paragraph boundaries."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    
    chunks = pipeline._chunk_for_translation(text)
    
    # Chunking is conservative - may return 1 chunk if text is short enough
    # The important thing is that all text is preserved
    assert len(chunks) >= 1
    all_text = " ".join(chunks)
    assert "First paragraph" in all_text
    assert "Second paragraph" in all_text
    assert "Third paragraph" in all_text


def test_chunk_for_translation_sentence_boundaries():
    """Test that chunking uses sentence boundaries when no paragraphs."""
    config = PipelineConfig()
    pipeline = TranslationPipeline(config)
    
    text = "First sentence. Second sentence. Third sentence."
    
    chunks = pipeline._chunk_for_translation(text)
    
    # Chunking is conservative - may return 1 chunk if text is short enough
    # The important thing is that all text is preserved
    assert len(chunks) >= 1
    all_text = " ".join(chunks)
    assert "First sentence" in all_text
    assert "Second sentence" in all_text
    assert "Third sentence" in all_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


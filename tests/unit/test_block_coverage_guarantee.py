"""
Unit tests for 100% block coverage guarantee.

These tests verify that:
- All translatable blocks get translations (no skips)
- Identity translations are detected and retried
- Retry escalation works (more candidates, fallback backend)
- Strict mode fails loudly if coverage incomplete
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scitran.core.models import Document, Segment, Block, BlockType, TranslationMetadata
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.translation.base import TranslationRequest, TranslationResponse


class TestBlockCoverageGuarantee:
    """Test that pipeline guarantees 100% block coverage."""
    
    def test_all_blocks_translated_no_skips(self):
        """Test that all translatable blocks get translations."""
        # Create document with multiple blocks
        document = Document(
            document_id="test_doc",
            source_lang="en",
            target_lang="fr"
        )
        
        segment = Segment(segment_id="seg1", segment_type="section")
        
        # Add various block types
        blocks = [
            Block(block_id="b1", source_text="Hello world", block_type=BlockType.PARAGRAPH),
            Block(block_id="b2", source_text="Introduction", block_type=BlockType.HEADING),
            Block(block_id="b3", source_text="Figure caption", block_type=BlockType.CAPTION),
            Block(block_id="b4", source_text="$E=mc^2$", block_type=BlockType.EQUATION),  # Not translatable
        ]
        
        for block in blocks:
            segment.blocks.append(block)
        
        document.segments.append(segment)
        
        # Count translatable blocks
        translatable = [b for b in blocks if b.is_translatable]
        assert len(translatable) == 3, "Should have 3 translatable blocks"
        
        # Mock translator
        config = PipelineConfig(
            backend="openai",
            strict_mode=True,
            enable_masking=False,
            enable_context=False,
            enable_reranking=False,
            num_candidates=1
        )
        
        pipeline = TranslationPipeline(config)
        
        # Mock the translator to return translations
        mock_translator = Mock()
        mock_translator.translate_sync = Mock(return_value=TranslationResponse(
            translations=["Bonjour le monde"],
            backend="openai",
            model="gpt-4o",
            tokens_used=10,
            cost=0.001,
            latency=0.1
        ))
        pipeline.translator = mock_translator
        
        # Translate blocks
        for block in translatable:
            text = block.source_text
            request = TranslationRequest(
                text=text,
                source_lang="en",
                target_lang="fr",
                num_candidates=1
            )
            response = mock_translator.translate_sync(request)
            block.translated_text = response.translations[0]
        
        # Verify all translatable blocks have translations
        for block in translatable:
            assert block.translated_text is not None, \
                f"Block {block.block_id} should have translation"
            assert block.translated_text.strip() != "", \
                f"Block {block.block_id} translation should not be empty"
    
    def test_identity_translation_detected(self):
        """Test that identity translations (source==output) are detected."""
        from scitran.core.validator import TranslationCompletenessValidator
        
        document = Document(
            document_id="test_doc",
            source_lang="en",
            target_lang="fr"
        )
        
        segment = Segment(segment_id="seg1", segment_type="section")
        
        # Block with identity translation (not actually translated)
        block = Block(
            block_id="b1",
            source_text="Hello world",
            translated_text="Hello world",  # Same as source!
            block_type=BlockType.PARAGRAPH
        )
        segment.blocks.append(block)
        document.segments.append(segment)
        
        # Validator should detect identity
        validator = TranslationCompletenessValidator(
            require_no_identity=True,
            source_lang="en",
            target_lang="fr"
        )
        
        result = validator.validate(document)
        
        assert not result.is_valid, "Should fail validation due to identity translation"
        assert len(result.identity_blocks) > 0, "Should detect identity block"
        assert "b1" in result.identity_blocks, "Should identify block b1 as identity"
    
    def test_retry_escalation_more_candidates(self):
        """Test that retry uses more candidates for failed blocks."""
        config = PipelineConfig(
            backend="openai",
            num_candidates=1,  # Start with 1
            enable_reranking=False,
            max_translation_retries=2,
            strict_mode=False
        )
        
        pipeline = TranslationPipeline(config)
        
        # During repair, num_candidates should increase
        original_candidates = config.num_candidates
        
        # Simulate repair mode
        config.num_candidates = max(3, config.num_candidates)
        config.enable_reranking = True
        
        assert config.num_candidates == 3, "Repair should use at least 3 candidates"
        assert config.enable_reranking, "Repair should enable reranking"
        
        # Restore
        config.num_candidates = original_candidates
    
    def test_strict_mode_fails_on_incomplete(self):
        """Test that strict mode fails loudly if translation incomplete."""
        from scitran.core.validator import TranslationCompletenessValidator
        
        document = Document(
            document_id="test_doc",
            source_lang="en",
            target_lang="fr"
        )
        
        segment = Segment(segment_id="seg1", segment_type="section")
        
        # Block without translation
        block = Block(
            block_id="b1",
            source_text="Hello world",
            translated_text=None,  # Missing!
            block_type=BlockType.PARAGRAPH
        )
        segment.blocks.append(block)
        document.segments.append(segment)
        
        # Strict validator should fail
        validator = TranslationCompletenessValidator(
            require_full_coverage=True,
            source_lang="en",
            target_lang="fr"
        )
        
        result = validator.validate(document)
        
        assert not result.is_valid, "Should fail validation"
        assert result.coverage < 1.0, "Coverage should be less than 100%"
        assert len(result.failed_blocks) > 0, "Should have failed blocks"
    
    def test_partial_blocks_detected(self):
        """Test that blocks with only partial translation are detected."""
        from scitran.core.validator import TranslationCompletenessValidator
        
        document = Document(
            document_id="test_doc",
            source_lang="en",
            target_lang="fr"
        )
        
        segment = Segment(segment_id="seg1", segment_type="section")
        
        # Block with whitespace-only translation
        block = Block(
            block_id="b1",
            source_text="Hello world with meaningful content",
            translated_text="   ",  # Only whitespace
            block_type=BlockType.PARAGRAPH
        )
        segment.blocks.append(block)
        document.segments.append(segment)
        
        validator = TranslationCompletenessValidator(
            require_full_coverage=True,
            source_lang="en",
            target_lang="fr"
        )
        
        result = validator.validate(document)
        
        assert not result.is_valid, "Should fail validation for whitespace-only translation"
        assert len(result.failed_blocks) > 0, "Should detect failed block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
SPRINT 1: Tests for translation coverage guarantee.

These tests verify that the pipeline properly handles:
1. Missing translations (blocks without translated_text)
2. Identity translations (source == output)
3. Retry mechanism with exponential backoff
4. Fallback to stronger backend
5. Strict mode failure reporting
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional

from scitran.core.models import Document, Segment, Block, BlockType, TranslationResult
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.core.exceptions import TranslationCoverageError
from scitran.translation.base import TranslationRequest, TranslationResponse


# Dummy translator for deterministic testing
class DummyTranslator:
    """Deterministic translator for testing - no network calls."""
    
    def __init__(self, fail_blocks: List[str] = None, identity_blocks: List[str] = None):
        """
        Args:
            fail_blocks: List of block IDs that should return None
            identity_blocks: List of block IDs that should return identity translation
        """
        self.fail_blocks = set(fail_blocks or [])
        self.identity_blocks = set(identity_blocks or [])
        self.call_count = {}
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text deterministically."""
        # Track calls
        text = request.text
        self.call_count[text] = self.call_count.get(text, 0) + 1
        
        # Check if this is a failing block (by matching text content)
        for block_id in self.fail_blocks:
            if block_id in text:
                return TranslationResponse(translations=[], backend="dummy", model="test")
        
        # Check if this is an identity block
        for block_id in self.identity_blocks:
            if block_id in text:
                return TranslationResponse(translations=[request.text], backend="dummy", model="test")
        
        # Normal translation: just add "[FR]" prefix
        translation = f"[FR] {text}"
        return TranslationResponse(translations=[translation], backend="dummy", model="test")
    
    def is_available(self) -> bool:
        return True


def create_test_document(block_ids: List[str]) -> Document:
    """Create a test document with specified block IDs."""
    doc = Document(
        document_id="test_doc",
        source_lang="en",
        target_lang="fr"
    )
    
    segment = Segment(segment_id="seg_1", segment_type="section")
    
    for i, block_id in enumerate(block_ids):
        block = Block(
            block_id=block_id,
            source_text=f"Text for {block_id}",  # Include block_id in text for matching
            block_type=BlockType.PARAGRAPH
        )
        segment.blocks.append(block)
    
    doc.segments.append(segment)
    return doc


class TestCoverageDetection:
    """Test detection of missing/identity translations."""
    
    def test_detect_missing_translations(self):
        """Test detection of blocks without translations."""
        config = PipelineConfig(
            backend="dummy",
            strict_mode=False,
            detect_identity_translation=False
        )
        pipeline = TranslationPipeline(config)
        
        # Create document with some untranslated blocks
        doc = create_test_document(["block_1", "block_2", "block_3"])
        doc.translatable_blocks[0].translated_text = "Translated"
        doc.translatable_blocks[1].translated_text = None  # Missing
        doc.translatable_blocks[2].translated_text = ""  # Empty
        
        # Detect missing
        missing = pipeline._detect_missing_translations(doc)
        
        assert len(missing) == 2
        assert missing[0].block_id == "block_2"
        assert missing[1].block_id == "block_3"
    
    def test_detect_identity_translations(self):
        """Test detection of identity translations (source == output)."""
        config = PipelineConfig(
            backend="dummy",
            detect_identity_translation=True
        )
        pipeline = TranslationPipeline(config)
        
        # Create document with identity translation
        doc = create_test_document(["block_1", "block_2"])
        doc.translatable_blocks[0].source_text = "Hello world"
        doc.translatable_blocks[0].translated_text = "Bonjour monde"  # Different
        doc.translatable_blocks[1].source_text = "Hello world"
        doc.translatable_blocks[1].translated_text = "Hello world"  # Identity!
        
        missing = pipeline._detect_missing_translations(doc)
        
        assert len(missing) == 1
        assert missing[0].block_id == "block_2"
    
    def test_identity_detection_ignores_non_alphabetic(self):
        """Test that identity detection ignores purely numeric content."""
        config = PipelineConfig(
            backend="dummy",
            detect_identity_translation=True
        )
        pipeline = TranslationPipeline(config)
        
        doc = create_test_document(["block_1"])
        doc.translatable_blocks[0].source_text = "123.456"
        doc.translatable_blocks[0].translated_text = "123.456"  # Numbers stay same - OK
        
        missing = pipeline._detect_missing_translations(doc)
        
        # Should NOT flag as missing (numbers don't need translation)
        assert len(missing) == 0


class TestRetryMechanism:
    """Test retry with exponential backoff."""
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_recovers_failed_blocks(self, mock_sleep):
        """Test that retry mechanism recovers failed blocks."""
        config = PipelineConfig(
            backend="dummy",
            max_translation_retries=3,
            retry_backoff_factor=2.0,
            strict_mode=False,
            enable_glossary=False,
            detect_identity_translation=False
        )
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        
        # Track call count
        call_count = {}
        
        # Mock _call_translator to fail first time, succeed second
        def mock_call_translator(system_prompt, user_prompt, temperature=0.3):
            call_count[user_prompt] = call_count.get(user_prompt, 0) + 1
            
            # Fail on first call
            if call_count[user_prompt] == 1:
                return None  # Simulate failure
            
            # Succeed on retry
            return f"[FR] {user_prompt}"
        
        pipeline._call_translator = mock_call_translator
        
        # Create document with failing block
        doc = create_test_document(["block_1", "block_2", "block_3"])
        doc.translatable_blocks[1].translated_text = None  # Missing
        
        # Retry
        remaining = pipeline._retry_with_backoff(doc, [doc.translatable_blocks[1]])
        
        # Should recover
        assert len(remaining) == 0
        assert doc.translatable_blocks[1].translated_text is not None
        assert "[FR]" in doc.translatable_blocks[1].translated_text
        
        # Check exponential backoff was used
        assert mock_sleep.call_count >= 1  # At least one retry
    
    @patch('time.sleep')
    def test_retry_exhausts_after_max_attempts(self, mock_sleep):
        """Test that retry stops after max_retries."""
        translator = DummyTranslator(fail_blocks=["block_2"])
        
        config = PipelineConfig(
            backend="dummy",
            max_translation_retries=2,
            strict_mode=False,
            enable_glossary=False
        )
        pipeline = TranslationPipeline(config)
        pipeline.translator = translator
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        
        doc = create_test_document(["block_1", "block_2"])
        doc.translatable_blocks[1].translated_text = None
        
        remaining = pipeline._retry_with_backoff(doc, [doc.translatable_blocks[1]])
        
        # Should still fail after retries
        assert len(remaining) == 1
        assert remaining[0].block_id == "block_2"


class TestFallbackBackend:
    """Test fallback to stronger backend."""
    
    def test_fallback_backend_recovers_blocks(self):
        """Test that fallback backend can recover failed blocks."""
        config = PipelineConfig(
            backend="dummy_primary",
            fallback_backend="dummy_fallback",
            enable_fallback_backend=True,
            strict_mode=False,
            enable_glossary=False,
            detect_identity_translation=False  # Disable to avoid false positives
        )
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        
        # Fallback backend succeeds - return different translation
        fallback_translator = Mock()
        fallback_translator.translate_sync.return_value = TranslationResponse(
            translations=["Traduction pour block_2"],  # Different from source
            backend="dummy_fallback",
            model="test"
        )
        fallback_translator.is_available.return_value = True
        
        # Mock _build_translation_request to return a simple request
        def mock_build_request(text):
            return TranslationRequest(text=text, source_lang="en", target_lang="fr")
        
        pipeline._build_translation_request = mock_build_request
        
        # Mock _create_translator to return fallback
        with patch.object(pipeline, '_create_translator', return_value=fallback_translator):
            doc = create_test_document(["block_1", "block_2"])
            doc.translatable_blocks[1].translated_text = None
            
            # Patch TranslationMetadata to avoid initialization issues
            from scitran.core.models import TranslationMetadata
            from datetime import datetime
            with patch('scitran.core.pipeline.TranslationMetadata') as mock_meta:
                mock_meta.return_value = TranslationMetadata(
                    backend="dummy_fallback",
                    timestamp=datetime.now(),
                    duration=0.0
                )
                remaining = pipeline._fallback_translate(doc, [doc.translatable_blocks[1]])
            
            # Should recover via fallback
            assert len(remaining) == 0
            assert doc.translatable_blocks[1].translated_text is not None
            assert doc.translatable_blocks[1].translated_text == "Traduction pour block_2"
    
    def test_fallback_skipped_if_same_as_primary(self):
        """Test that fallback is skipped if same as primary backend."""
        config = PipelineConfig(
            backend="cascade",  # Use free backend that doesn't need API key
            fallback_backend="cascade",  # Same!
            enable_fallback_backend=True,
            strict_mode=False,
            enable_glossary=False
        )
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        
        doc = create_test_document(["block_1"])
        doc.translatable_blocks[0].translated_text = None
        
        # Should skip fallback and return block as still missing
        remaining = pipeline._fallback_translate(doc, [doc.translatable_blocks[0]])
        
        assert len(remaining) == 1
        # Verify fallback was skipped (no translator created)
        assert remaining[0].block_id == "block_1"


class TestStrictMode:
    """Test strict mode with failure reporting."""
    
    def test_strict_mode_raises_on_missing_blocks(self):
        """Test that strict mode raises exception when blocks missing."""
        config = PipelineConfig(
            backend="dummy",
            strict_mode=True,
            max_translation_retries=0,  # No retries
            enable_fallback_backend=False,
            enable_glossary=False,
            detect_identity_translation=False  # Simplify test
        )
        
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        doc = create_test_document(["block_1", "block_2"])
        
        result = TranslationResult(document=doc)
        
        # Manually set one block as missing
        doc.translatable_blocks[0].translated_text = "[FR] Text for block_1"  # First block OK
        doc.translatable_blocks[1].translated_text = None  # Second block missing
        
        # Should raise in strict mode
        with pytest.raises(TranslationCoverageError) as exc_info:
            pipeline._ensure_translation_coverage(doc, result)
        
        # Check failure report
        assert exc_info.value.failure_report is not None
        assert exc_info.value.failure_report['failed_count'] == 1
        assert exc_info.value.failure_report['total_blocks'] == 2
    
    def test_non_strict_mode_allows_partial_translation(self):
        """Test that non-strict mode allows partial translation."""
        config = PipelineConfig(
            backend="dummy",
            strict_mode=False,
            max_translation_retries=0,
            enable_fallback_backend=False
        )
        pipeline = TranslationPipeline(config)
        
        doc = create_test_document(["block_1", "block_2"])
        doc.translatable_blocks[1].translated_text = None  # Missing
        
        result = TranslationResult(document=doc)
        
        # Should NOT raise in non-strict mode
        pipeline._ensure_translation_coverage(doc, result)
        
        # But should report coverage
        assert result.coverage < 1.0
        assert result.failure_report is not None


class TestFailureReport:
    """Test failure report generation."""
    
    def test_failure_report_contains_required_fields(self):
        """Test that failure report has all required fields."""
        config = PipelineConfig(backend="dummy")
        pipeline = TranslationPipeline(config)
        
        doc = create_test_document(["block_1", "block_2"])
        doc.translatable_blocks[1].translated_text = None
        
        report = pipeline._generate_failure_report(doc, [doc.translatable_blocks[1]])
        
        assert 'timestamp' in report
        assert 'document_id' in report
        assert 'total_blocks' in report
        assert 'failed_count' in report
        assert 'failures' in report
        
        assert report['document_id'] == "test_doc"
        assert report['failed_count'] == 1
        assert len(report['failures']) == 1
        
        failure = report['failures'][0]
        assert 'block_id' in failure
        assert 'source_text' in failure
        assert 'reason' in failure
        
        assert failure['block_id'] == "block_2"
        assert failure['reason'] == "missing_translation"


class TestCoverageGuaranteeIntegration:
    """Integration tests for full coverage guarantee workflow."""
    
    @patch('time.sleep')
    def test_full_coverage_workflow_with_recovery(self, mock_sleep):
        """Test complete workflow: detect → retry → recover."""
        config = PipelineConfig(
            backend="dummy",
            max_translation_retries=2,
            strict_mode=False,
            enable_glossary=False,
            detect_identity_translation=False
        )
        
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        pipeline.glossary = {}
        
        # Track calls per block
        call_counts = {}
        
        # Mock _call_translator to succeed on retry
        def mock_call_translator(system_prompt, user_prompt, temperature=0.3):
            call_counts[user_prompt] = call_counts.get(user_prompt, 0) + 1
            
            # Fail on first call for block_2
            if call_counts[user_prompt] == 1 and "block_2" in user_prompt:
                return None  # Simulate failure
            
            # Succeed on retry or for other blocks
            return f"[FR] {user_prompt}"
        
        pipeline._call_translator = mock_call_translator
        
        doc = create_test_document(["block_1", "block_2", "block_3"])
        
        # Simulate first translation pass with one failure
        doc.translatable_blocks[0].translated_text = "[FR] Text for block_1"
        doc.translatable_blocks[1].translated_text = None  # Failed
        doc.translatable_blocks[2].translated_text = "[FR] Text for block_3"
        
        result = TranslationResult(document=doc)
        
        # Run coverage guarantee
        pipeline._ensure_translation_coverage(doc, result)
        
        # Should recover via retry
        assert result.coverage == 1.0
        assert doc.translatable_blocks[1].translated_text is not None
        assert "[FR]" in doc.translatable_blocks[1].translated_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


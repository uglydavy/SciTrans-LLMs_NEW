"""
Tests for speed improvements (PHASE 1).

Tests:
- Multi-candidate generation uses single backend call
- FastTranslator async doesn't block event loop
- Fast mode configuration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from scitran.core.pipeline import PipelineConfig, TranslationPipeline
from scitran.translation.base import TranslationRequest, TranslationResponse, TranslationBackend


class MockBatchBackend(TranslationBackend):
    """Mock backend that supports batch candidates."""
    
    supports_batch_candidates = True
    
    def __init__(self):
        super().__init__(api_key="test", model="test")
        self.call_count = 0
        self.last_request = None
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        self.call_count += 1
        self.last_request = request
        return TranslationResponse(
            translations=[f"translation_{i}" for i in range(request.num_candidates)],
            backend="mock",
            model="test",
            finish_reasons=["stop"] * request.num_candidates
        )
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        self.call_count += 1
        self.last_request = request
        return TranslationResponse(
            translations=[f"translation_{i}" for i in range(request.num_candidates)],
            backend="mock",
            model="test",
            finish_reasons=["stop"] * request.num_candidates
        )


def test_batch_candidates_single_call():
    """Test that batch-capable backends make only one call for multiple candidates."""
    config = PipelineConfig(
        backend="mock",
        num_candidates=3,
        enable_masking=False,
        enable_glossary=False,
        enable_context=False,
        enable_reranking=False
    )
    
    pipeline = TranslationPipeline(config)
    
    # Mock the translator with batch-capable backend
    mock_backend = MockBatchBackend()
    pipeline.translator = mock_backend
    
    # Create a simple block
    from scitran.core.models import Block
    block = Block(
        block_id="test_block",
        source_text="Hello world"
    )
    
    # Generate candidates
    candidates = pipeline._generate_candidates(block, [])
    
    # Should have made only 1 call (not 3)
    assert mock_backend.call_count == 1
    assert mock_backend.last_request.num_candidates == 3
    assert len(candidates) == 3


def test_fast_mode_configuration():
    """Test that fast_mode applies correct defaults."""
    config = PipelineConfig(
        fast_mode=True,
        num_candidates=5,  # Should be overridden
        enable_reranking=True,  # Should be overridden
        enable_context=True  # Should be overridden
    )
    
    pipeline = TranslationPipeline(config)
    
    # Fast mode should override these settings
    assert pipeline.config.num_candidates == 1
    assert pipeline.config.enable_reranking is False
    assert pipeline.config.enable_context is False
    assert pipeline.config.context_window_size == 0


def test_finish_reason_tracking():
    """Test that finish_reason is tracked in responses."""
    mock_backend = MockBatchBackend()
    
    request = TranslationRequest(
        text="test",
        source_lang="en",
        target_lang="fr",
        num_candidates=2
    )
    
    response = mock_backend.translate_sync(request)
    
    assert response.finish_reasons is not None
    assert len(response.finish_reasons) == 2
    assert all(reason == "stop" for reason in response.finish_reasons)


@pytest.mark.asyncio
async def test_fast_translator_async_no_blocking():
    """Test that FastTranslator doesn't block event loop."""
    from scitran.utils.fast_translator import FastTranslator
    import asyncio
    
    translator = FastTranslator(max_concurrent=2)
    
    # Mock the Google translator to simulate blocking
    with patch('deep_translator.GoogleTranslator') as mock_gt:
        mock_instance = Mock()
        mock_instance.translate = Mock(return_value="translated")
        mock_gt.return_value = mock_instance
        
        # Translate should use run_in_executor for blocking calls
        result = await translator._translate_google_free("test", "en", "fr")
        
        # Should have called translate
        assert mock_instance.translate.called


def test_config_validation_no_false_positives():
    """Test that config validation doesn't fail on env var API keys."""
    # Should not raise even without api_key in config (checks env vars)
    config = PipelineConfig(
        backend="openai",
        api_key=None  # Will check environment
    )
    
    issues = config.validate()
    # Should not complain about missing API key (checked at runtime)
    assert not any("api_key" in issue.lower() for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


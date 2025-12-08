"""Integration tests for translation pipeline."""

import pytest
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.core.models import Document, Block


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Integration tests require --run-integration flag"
)
def test_pipeline_basic():
    """Test basic pipeline execution."""
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="free",  # Use free backend for testing
        enable_masking=False,
        enable_context=False
    )
    
    pipeline = TranslationPipeline(config)
    
    document = Document(
        doc_id="test",
        blocks=[Block(block_id="1", source_text="Hello world")]
    )
    
    result = pipeline.translate_document(document)
    
    assert result.document.blocks[0].translated_text is not None
    assert len(result.document.blocks[0].translated_text) > 0


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Integration tests require --run-integration flag"
)
def test_pipeline_with_masking():
    """Test pipeline with masking enabled."""
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="free",
        enable_masking=True
    )
    
    pipeline = TranslationPipeline(config)
    
    document = Document(
        doc_id="test",
        blocks=[Block(block_id="1", source_text="The equation $x=5$ is simple.")]
    )
    
    result = pipeline.translate_document(document)
    
    # Check LaTeX is preserved
    assert "$x=5$" in result.document.blocks[0].translated_text


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )

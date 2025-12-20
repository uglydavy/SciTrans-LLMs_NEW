"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The equation $E = mc^2$ is fundamental in physics."

@pytest.fixture
def sample_block():
    """Sample block for testing."""
    from scitran.core.models import Block
    return Block(
        block_id="test_1",
        source_text="Machine learning enables state-of-the-art performance."
    )

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    from scitran.core.models import Document, Block
    return Document(
        doc_id="test_doc",
        blocks=[
            Block(block_id="1", source_text="First paragraph."),
            Block(block_id="2", source_text="Second paragraph with $x = y + z$."),
            Block(block_id="3", source_text="Third paragraph.")
        ]
    )

@pytest.fixture
def mock_config():
    """Mock pipeline configuration."""
    from scitran.core.pipeline import PipelineConfig
    return PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai",
        enable_masking=True,
        enable_context=True
    )

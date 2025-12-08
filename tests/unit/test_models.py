"""Unit tests for core models."""

import pytest
from scitran.core.models import Block, Document, BoundingBox


def test_block_creation():
    """Test Block creation."""
    block = Block(
        block_id="test_1",
        source_text="Test text"
    )
    
    assert block.block_id == "test_1"
    assert block.source_text == "Test text"
    assert block.translated_text is None


def test_block_with_bbox():
    """Test Block with bounding box."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, page=0)
    block = Block(
        block_id="test_2",
        source_text="Text with position",
        bbox=bbox
    )
    
    assert block.bbox.x == 10
    assert block.bbox.page == 0


def test_document_creation():
    """Test Document creation."""
    blocks = [
        Block(block_id="1", source_text="First"),
        Block(block_id="2", source_text="Second")
    ]
    
    doc = Document(doc_id="test_doc", blocks=blocks)
    
    assert doc.doc_id == "test_doc"
    assert len(doc.blocks) == 2


def test_document_serialization():
    """Test Document serialization."""
    block = Block(block_id="1", source_text="Test")
    doc = Document(doc_id="test", blocks=[block])
    
    data = doc.to_dict()
    
    assert data["doc_id"] == "test"
    assert len(data["blocks"]) == 1
    assert data["blocks"][0]["source_text"] == "Test"


def test_document_deserialization():
    """Test Document deserialization."""
    data = {
        "doc_id": "test",
        "blocks": [
            {"block_id": "1", "source_text": "Test"}
        ]
    }
    
    doc = Document.from_dict(data)
    
    assert doc.doc_id == "test"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].source_text == "Test"

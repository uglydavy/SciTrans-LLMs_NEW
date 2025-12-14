"""Unit tests for core models."""

import pytest
from scitran.core.models import Block, Document, BoundingBox, Segment
from dataclasses import asdict


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
    bbox = BoundingBox(page=0, x0=10.0, y0=20.0, x1=110.0, y1=70.0)
    block = Block(
        block_id="test_2",
        source_text="Text with position",
        bbox=bbox
    )
    
    assert block.bbox.x0 == 10.0
    assert block.bbox.page == 0


def test_document_creation():
    """Test Document creation."""
    blocks = [
        Block(block_id="1", source_text="First"),
        Block(block_id="2", source_text="Second")
    ]
    segment = Segment(segment_id="seg1", segment_type="section", blocks=blocks)
    doc = Document(document_id="test_doc", segments=[segment])
    
    assert doc.document_id == "test_doc"
    assert len(doc.all_blocks) == 2


def test_document_serialization():
    """Test Document serialization."""
    block = Block(block_id="1", source_text="Test")
    segment = Segment(segment_id="seg1", segment_type="section", blocks=[block])
    doc = Document(document_id="test", segments=[segment])
    
    data = asdict(doc)
    
    assert data["document_id"] == "test"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["blocks"][0]["source_text"] == "Test"


def test_document_deserialization():
    """Test Document creation from components."""
    segment_data = Segment(
        segment_id="seg1",
        segment_type="section",
        blocks=[Block(block_id="1", source_text="Test")]
    )
    
    doc = Document(
        document_id="test",
        segments=[segment_data]
    )
    
    assert doc.document_id == "test"
    assert len(doc.all_blocks) == 1
    assert doc.all_blocks[0].source_text == "Test"

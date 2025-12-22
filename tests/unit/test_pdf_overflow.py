"""
Tests for PDF rendering overflow handling (PHASE 3.2).

Tests:
- Overflow detection
- Overflow strategies
- Overflow report generation
"""

import pytest
from pathlib import Path
from scitran.rendering.pdf_renderer import PDFRenderer
from scitran.core.models import Document, Segment, Block, BoundingBox, FontInfo, BlockType


def create_test_block(text: str, page: int = 0) -> Block:
    """Create a test block with bbox."""
    return Block(
        block_id="test_block",
        source_text=text,
        translated_text=text,
        bbox=BoundingBox(x0=50, y0=50, x1=150, y1=80, page=page),
        font=FontInfo(family="Arial", size=10)
    )


def test_overflow_report_creation():
    """Test that overflow events are tracked."""
    renderer = PDFRenderer(
        overflow_strategy="shrink",
        min_font_size=4.0
    )
    
    # Initially empty
    assert len(renderer.overflow_report) == 0


def test_overflow_strategy_shrink():
    """Test shrink strategy configuration."""
    renderer = PDFRenderer(
        overflow_strategy="shrink",
        min_font_size=3.0,
        min_lineheight=0.9
    )
    
    assert renderer.overflow_strategy == "shrink"
    assert renderer.min_font_size == 3.0
    assert renderer.min_lineheight == 0.9


def test_overflow_strategy_expand():
    """Test expand strategy configuration."""
    renderer = PDFRenderer(
        overflow_strategy="expand"
    )
    
    assert renderer.overflow_strategy == "expand"


def test_overflow_strategy_append_pages():
    """Test append_pages strategy configuration."""
    renderer = PDFRenderer(
        overflow_strategy="append_pages"
    )
    
    assert renderer.overflow_strategy == "append_pages"


def test_overflow_strategy_marker_append():
    """Test marker+append_pages strategy configuration."""
    renderer = PDFRenderer(
        overflow_strategy="marker+append_pages"
    )
    
    assert renderer.overflow_strategy == "marker+append_pages"


def test_overflow_report_save(tmp_path):
    """Test that overflow report is saved to JSON."""
    renderer = PDFRenderer()
    
    # Add some overflow events
    renderer.overflow_report.append({
        "block_id": "block_1",
        "page": 0,
        "text_length": 500,
        "strategy": "shrink"
    })
    renderer.overflow_report.append({
        "block_id": "block_2",
        "page": 1,
        "text_length": 300,
        "strategy": "shrink"
    })
    
    # Save report
    output_path = str(tmp_path / "test.pdf")
    renderer.save_overflow_report(output_path)
    
    # Check report file exists
    report_path = tmp_path / "test_overflow_report.json"
    assert report_path.exists()
    
    # Check report content
    import json
    with open(report_path) as f:
        report = json.load(f)
    
    assert report["total_overflows"] == 2
    assert report["strategy"] == "shrink"
    assert len(report["events"]) == 2


def test_no_overflow_report_when_empty(tmp_path):
    """Test that no report is saved when there are no overflows."""
    renderer = PDFRenderer()
    
    # No overflow events
    assert len(renderer.overflow_report) == 0
    
    # Save report (should do nothing)
    output_path = str(tmp_path / "test.pdf")
    renderer.save_overflow_report(output_path)
    
    # Check report file doesn't exist
    report_path = tmp_path / "test_overflow_report.json"
    assert not report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


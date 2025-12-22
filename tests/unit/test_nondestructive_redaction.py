"""
Unit tests for non-destructive redaction.

These tests verify that redaction removes text without damaging:
- Vector graphics (lines, rectangles, curves)
- Embedded images
- Table borders
- Figure diagrams
"""

import pytest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from tests.fixtures.pdf_generator import TestPDFGenerator
from scitran.rendering.pdf_renderer import PDFRenderer
from scitran.core.models import Document, Segment, Block, BoundingBox, BlockType


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestNonDestructiveRedaction:
    """Test that redaction doesn't damage graphics."""
    
    def test_redaction_preserves_vector_line(self):
        """Test that redacting text near a vector line doesn't erase the line."""
        # Create test PDF with red line and text
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_vector_and_text()
        
        try:
            # Verify the source PDF has the red line
            source_doc = fitz.open(test_pdf)
            source_page = source_doc[0]
            source_drawings = source_page.get_drawings()
            
            # Should have at least 2 drawings (red line + blue rectangle)
            assert len(source_drawings) >= 2, "Source PDF should have vector drawings"
            
            # Check for red color in drawings
            has_red = any(
                d.get('color') and d['color'][0] > 0.9  # Red component
                for d in source_drawings
            )
            assert has_red, "Source PDF should have red vector graphics"
            source_doc.close()
            
            # Create a simple document with blocks to redact
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Block that overlaps the red line (text at 150, 200, 350, 250)
            block = Block(
                block_id="block1",
                source_text="This text overlaps the red line",
                translated_text="Ce texte chevauche la ligne rouge",
                bbox=BoundingBox(x0=150, y0=200, x1=350, y1=250, page=0),
                block_type=BlockType.PARAGRAPH
            )
            segment.blocks.append(block)
            document.segments.append(segment)
            
            # Redact and render
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify the output PDF still has the red line
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            
            # Should still have drawings
            assert len(output_drawings) >= 2, \
                f"Output PDF should preserve vector drawings (found {len(output_drawings)})"
            
            # Check for red color still present
            has_red_after = any(
                d.get('color') and d['color'][0] > 0.9
                for d in output_drawings
            )
            assert has_red_after, \
                "Output PDF should still have red vector graphics after redaction"
            
            output_doc.close()
            
            # Cleanup
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)
    
    def test_redaction_preserves_rectangle(self):
        """Test that redacting text near a rectangle doesn't erase the rectangle."""
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_vector_and_text()
        
        try:
            # Create document with block near the blue rectangle
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Block near blue rectangle (text at 50, 620, 200, 650)
            block = Block(
                block_id="block2",
                source_text="Text near blue box",
                translated_text="Texte près de la boîte bleue",
                bbox=BoundingBox(x0=50, y0=620, x1=200, y1=650, page=0),
                block_type=BlockType.PARAGRAPH
            )
            segment.blocks.append(block)
            document.segments.append(segment)
            
            # Render
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify blue rectangle still exists
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            
            has_blue = any(
                d.get('color') and d['color'][2] > 0.9  # Blue component
                for d in output_drawings
            )
            assert has_blue, "Output PDF should still have blue rectangle after redaction"
            
            output_doc.close()
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)
    
    def test_table_borders_preserved(self):
        """Test that table borders remain intact when table text is translated."""
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_table()
        
        try:
            # Count lines in source
            source_doc = fitz.open(test_pdf)
            source_page = source_doc[0]
            source_drawings = source_page.get_drawings()
            num_source_lines = len([d for d in source_drawings if 'items' in d])
            source_doc.close()
            
            assert num_source_lines >= 5, "Table should have multiple border lines"
            
            # Create document with table cell text blocks
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Add blocks for each cell
            cells = [
                ("block1", "Header 1", "En-tête 1", (55, 105, 195, 145)),
                ("block2", "Header 2", "En-tête 2", (205, 105, 345, 145)),
                ("block3", "Data 1", "Données 1", (55, 155, 195, 195)),
            ]
            
            for block_id, source, target, bbox in cells:
                block = Block(
                    block_id=block_id,
                    source_text=source,
                    translated_text=target,
                    bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page=0),
                    block_type=BlockType.PARAGRAPH  # Even though in table
                )
                segment.blocks.append(block)
            
            document.segments.append(segment)
            
            # Render
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify table lines still exist
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            num_output_lines = len([d for d in output_drawings if 'items' in d])
            output_doc.close()
            
            # Should have same number of lines (or close - some tolerance for rendering variations)
            assert num_output_lines >= num_source_lines - 1, \
                f"Table borders should be preserved (source: {num_source_lines}, output: {num_output_lines})"
            
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)
    
    def test_figure_graphics_preserved(self):
        """Test that figure graphics (graph lines) are preserved."""
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_figure()
        
        try:
            # Count drawings in source
            source_doc = fitz.open(test_pdf)
            source_page = source_doc[0]
            source_drawings = source_page.get_drawings()
            num_source_drawings = len(source_drawings)
            source_doc.close()
            
            assert num_source_drawings >= 7, "Figure should have multiple line segments"
            
            # Create document with caption block only
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Only translate the caption
            block = Block(
                block_id="caption",
                source_text="Figure 1: Sample graph showing data trend",
                translated_text="Figure 1 : Graphique d'exemple montrant la tendance des données",
                bbox=BoundingBox(x0=100, y0=420, x1=400, y1=450, page=0),
                block_type=BlockType.CAPTION
            )
            segment.blocks.append(block)
            document.segments.append(segment)
            
            # Render
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify figure graphics still exist
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            num_output_drawings = len(output_drawings)
            output_doc.close()
            
            # Should preserve most drawings (allow small variation)
            assert num_output_drawings >= num_source_drawings - 1, \
                f"Figure graphics should be preserved (source: {num_source_drawings}, output: {num_output_drawings})"
            
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


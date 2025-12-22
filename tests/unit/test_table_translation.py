"""
Unit tests for table/figure text translation policy.

These tests verify that:
- Table text can be translated when translate_table_text=True
- Figure text can be translated when translate_figure_text=True
- Table borders remain intact after translation
- Figure graphics remain intact after translation
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
class TestTableTranslation:
    """Test that table text can be translated while preserving borders."""
    
    def test_table_text_translated_borders_preserved(self):
        """Test that table cell text is translated while borders remain intact."""
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_table()
        
        try:
            # Count table borders in source
            source_doc = fitz.open(test_pdf)
            source_page = source_doc[0]
            source_drawings = source_page.get_drawings()
            num_source_lines = len([d for d in source_drawings if 'items' in d])
            source_doc.close()
            
            # Create document with table cells marked as TABLE type
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Add blocks for table cells (now with TABLE type)
            cells = [
                ("cell1", "Header 1", "En-tête 1", (55, 105, 195, 145)),
                ("cell2", "Header 2", "En-tête 2", (205, 105, 345, 145)),
                ("cell3", "Data 1", "Données 1", (55, 155, 195, 195)),
            ]
            
            for block_id, source, target, bbox in cells:
                block = Block(
                    block_id=block_id,
                    source_text=source,
                    translated_text=target,
                    bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page=0),
                    block_type=BlockType.TABLE  # Mark as TABLE
                )
                segment.blocks.append(block)
            
            document.segments.append(segment)
            
            # Render with translation
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify table borders still exist
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            num_output_lines = len([d for d in output_drawings if 'items' in d])
            
            # Check that borders are preserved
            assert num_output_lines >= num_source_lines - 1, \
                f"Table borders should be preserved (source: {num_source_lines}, output: {num_output_lines})"
            
            # Verify translated text is present
            text_dict = output_page.get_text("dict")
            page_text = ""
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            page_text += span.get("text", "") + " "
            
            # Should contain French text
            assert "En-tête" in page_text or "Données" in page_text, \
                "Translated text should be present in output"
            
            output_doc.close()
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)
    
    def test_figure_caption_translated_graphics_preserved(self):
        """Test that figure captions are translated while graphics remain intact."""
        generator = TestPDFGenerator()
        test_pdf = generator.create_pdf_with_figure()
        
        try:
            # Count figure graphics in source
            source_doc = fitz.open(test_pdf)
            source_page = source_doc[0]
            source_drawings = source_page.get_drawings()
            num_source_drawings = len(source_drawings)
            source_doc.close()
            
            # Create document with caption as CAPTION type
            document = Document(
                document_id="test_doc",
                source_lang="en",
                target_lang="fr"
            )
            
            segment = Segment(segment_id="seg1", segment_type="section")
            
            # Caption block
            block = Block(
                block_id="caption",
                source_text="Figure 1: Sample graph showing data trend",
                translated_text="Figure 1 : Graphique montrant la tendance",
                bbox=BoundingBox(x0=100, y0=420, x1=400, y1=450, page=0),
                block_type=BlockType.CAPTION
            )
            segment.blocks.append(block)
            document.segments.append(segment)
            
            # Render
            output_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            renderer = PDFRenderer()
            renderer.render_with_layout(str(test_pdf), document, str(output_pdf))
            
            # Verify figure graphics preserved
            output_doc = fitz.open(output_pdf)
            output_page = output_doc[0]
            output_drawings = output_page.get_drawings()
            num_output_drawings = len(output_drawings)
            
            assert num_output_drawings >= num_source_drawings - 1, \
                f"Figure graphics should be preserved (source: {num_source_drawings}, output: {num_output_drawings})"
            
            # Verify translated caption
            page_text = output_page.get_text()
            assert "Graphique" in page_text or "tendance" in page_text, \
                "Translated caption should be present"
            
            output_doc.close()
            output_pdf.unlink()
            
        finally:
            test_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


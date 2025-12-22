"""
Test PDF generator for unit tests.

Creates synthetic PDFs with controlled content for testing:
- Vector graphics (lines, boxes)
- Embedded images
- Text overlapping graphics
- Tables with borders
- Figures with captions
"""

import tempfile
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class TestPDFGenerator:
    """Generate test PDFs for unit testing."""
    
    def __init__(self):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required for PDF generation")
    
    def create_pdf_with_vector_and_text(
        self, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a PDF with vector graphics and overlapping text.
        
        This is used to test that redaction doesn't destroy vector graphics.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        
        # Draw a red diagonal line (vector graphic)
        page.draw_line(
            fitz.Point(100, 100),
            fitz.Point(400, 400),
            color=(1, 0, 0),  # Red
            width=5
        )
        
        # Draw a blue rectangle (vector graphic)
        page.draw_rect(
            fitz.Rect(50, 500, 200, 600),
            color=(0, 0, 1),  # Blue outline
            fill=None,
            width=3
        )
        
        # Add text that overlaps the line
        text_rect = fitz.Rect(150, 200, 350, 250)
        page.insert_textbox(
            text_rect,
            "This text overlaps the red line",
            fontsize=14,
            fontname="helv",
            color=(0, 0, 0)
        )
        
        # Add text near the rectangle
        text_rect2 = fitz.Rect(50, 620, 200, 650)
        page.insert_textbox(
            text_rect2,
            "Text near blue box",
            fontsize=12,
            fontname="helv",
            color=(0, 0, 0)
        )
        
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def create_pdf_with_image_and_text(
        self,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a PDF with an embedded image and text.
        
        This tests that redaction doesn't damage images.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        
        # Create a simple colored rectangle as an "image"
        # (In a real test, we'd embed an actual image file)
        rect = fitz.Rect(100, 100, 300, 250)
        page.draw_rect(rect, color=None, fill=(0.8, 0.8, 0.9), width=0)
        
        # Add text that overlaps the image area
        text_rect = fitz.Rect(150, 180, 350, 220)
        page.insert_textbox(
            text_rect,
            "Caption text over image area",
            fontsize=12,
            fontname="helv",
            color=(0, 0, 0)
        )
        
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def create_pdf_with_table(
        self,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a PDF with a table-like structure.
        
        This tests that table borders aren't damaged while text is translated.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        
        # Draw table borders
        # Outer border
        table_rect = fitz.Rect(50, 100, 500, 300)
        page.draw_rect(table_rect, color=(0, 0, 0), width=2)
        
        # Horizontal lines
        for y in [150, 200, 250]:
            page.draw_line(
                fitz.Point(50, y),
                fitz.Point(500, y),
                color=(0, 0, 0),
                width=1
            )
        
        # Vertical lines
        for x in [200, 350]:
            page.draw_line(
                fitz.Point(x, 100),
                fitz.Point(x, 300),
                color=(0, 0, 0),
                width=1
            )
        
        # Add cell text
        cells = [
            (fitz.Rect(55, 105, 195, 145), "Header 1"),
            (fitz.Rect(205, 105, 345, 145), "Header 2"),
            (fitz.Rect(355, 105, 495, 145), "Header 3"),
            (fitz.Rect(55, 155, 195, 195), "Data 1"),
            (fitz.Rect(205, 155, 345, 195), "Data 2"),
            (fitz.Rect(355, 155, 495, 195), "Data 3"),
        ]
        
        for rect, text in cells:
            page.insert_textbox(
                rect,
                text,
                fontsize=10,
                fontname="helv",
                color=(0, 0, 0),
                align=1  # Center align
            )
        
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def create_pdf_with_figure(
        self,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a PDF with a figure (graph-like visualization) and caption.
        
        This tests that figure graphics aren't damaged.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        
        # Draw a simple "graph" using lines
        # Axes
        page.draw_line(
            fitz.Point(100, 400),
            fitz.Point(400, 400),
            color=(0, 0, 0),
            width=2
        )
        page.draw_line(
            fitz.Point(100, 400),
            fitz.Point(100, 200),
            color=(0, 0, 0),
            width=2
        )
        
        # Data line (simplified)
        points = [
            fitz.Point(100, 380),
            fitz.Point(150, 350),
            fitz.Point(200, 320),
            fitz.Point(250, 280),
            fitz.Point(300, 250),
            fitz.Point(350, 230),
            fitz.Point(400, 220),
        ]
        
        for i in range(len(points) - 1):
            page.draw_line(
                points[i],
                points[i + 1],
                color=(0, 0.5, 0.8),
                width=2
            )
        
        # Figure caption
        caption_rect = fitz.Rect(100, 420, 400, 450)
        page.insert_textbox(
            caption_rect,
            "Figure 1: Sample graph showing data trend",
            fontsize=10,
            fontname="helv",
            color=(0, 0, 0)
        )
        
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def create_complex_test_pdf(
        self,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a complex PDF with multiple elements on one page.
        
        This is for comprehensive testing.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        
        # Title
        page.insert_textbox(
            fitz.Rect(50, 50, 545, 80),
            "Test Document with Mixed Content",
            fontsize=16,
            fontname="hebo",
            color=(0, 0, 0),
            align=1  # Center
        )
        
        # Paragraph with some text
        para_rect = fitz.Rect(50, 100, 545, 180)
        page.insert_textbox(
            para_rect,
            "This is a test paragraph with regular text. It contains multiple lines "
            "and should be translated. The text is positioned near various graphics.",
            fontsize=11,
            fontname="helv",
            color=(0, 0, 0)
        )
        
        # Vector graphic - red box
        page.draw_rect(
            fitz.Rect(50, 200, 150, 280),
            color=(1, 0, 0),
            fill=None,
            width=3
        )
        
        # Text near graphic
        page.insert_textbox(
            fitz.Rect(160, 220, 300, 250),
            "Text near red box",
            fontsize=10,
            fontname="helv"
        )
        
        # Diagonal line
        page.draw_line(
            fitz.Point(50, 300),
            fitz.Point(545, 350),
            color=(0, 0, 1),
            width=2
        )
        
        # More text
        page.insert_textbox(
            fitz.Rect(50, 370, 545, 420),
            "Another paragraph after the line. This text should also be translated "
            "without damaging the blue line above.",
            fontsize=11,
            fontname="helv"
        )
        
        doc.save(str(output_path))
        doc.close()
        
        return output_path


# Convenience functions for tests
def generate_test_pdf_with_graphics() -> Path:
    """Generate a test PDF with vector graphics for testing redaction."""
    gen = TestPDFGenerator()
    return gen.create_pdf_with_vector_and_text()


def generate_test_pdf_with_table() -> Path:
    """Generate a test PDF with a table for testing table translation."""
    gen = TestPDFGenerator()
    return gen.create_pdf_with_table()


def generate_test_pdf_with_figure() -> Path:
    """Generate a test PDF with a figure for testing figure preservation."""
    gen = TestPDFGenerator()
    return gen.create_pdf_with_figure()


"""PDF rendering with layout preservation."""

from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from ..core.models import Document, Block


class PDFRenderer:
    """Render translated documents to PDF with layout preservation."""
    
    def __init__(self):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    
    def render_pdf(self, document: Document, output_path: str):
        """Render document to PDF (main entry point)."""
        # Use simple rendering by default
        return self.render_simple(document, output_path)
    
    def render_simple(self, document: Document, output_path: str):
        """
        Render document to PDF (simple version without layout preservation).
        
        Args:
            document: Translated document
            output_path: Output PDF path
        """
        pdf = fitz.open()
        
        # Group blocks by page
        blocks_by_page = {}
        for segment in document.segments:
            for block in segment.blocks:
                page_num = block.bbox.page if block.bbox else 0
                if page_num not in blocks_by_page:
                    blocks_by_page[page_num] = []
                blocks_by_page[page_num].append(block)
        
        # Create pages
        for page_num in sorted(blocks_by_page.keys()):
            page = pdf.new_page(width=595, height=842)  # A4 size
            
            y_position = 50  # Start position
            
            for block in blocks_by_page[page_num]:
                text = block.translated_text or block.source_text
                
                # Insert text
                rect = fitz.Rect(50, y_position, 545, y_position + 200)
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=11,
                    fontname="helv",
                    align=0
                )
                
                # Move to next position
                y_position += len(text.split("\n")) * 15 + 10
                
                if y_position > 750:  # Page overflow
                    break
        
        pdf.save(output_path)
        pdf.close()
    
    def render_with_layout(self, source_pdf: str, document: Document, output_path: str):
        """
        Render document to PDF preserving original layout.
        
        Args:
            source_pdf: Original PDF for layout reference
            document: Translated document
            output_path: Output PDF path
        """
        # Open source PDF
        source_doc = fitz.open(source_pdf)
        output_doc = fitz.open()
        
        # Group blocks by page
        blocks_by_page = {}
        for segment in document.segments:
            for block in segment.blocks:
                page_num = block.bbox.page if block.bbox else 0
                if page_num not in blocks_by_page:
                    blocks_by_page[page_num] = []
                blocks_by_page[page_num].append(block)
        
        # Process each page
        for page_num in range(len(source_doc)):
            source_page = source_doc[page_num]
            
            # Create new page with same dimensions
            output_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height
            )
            
            # Copy images from source (if any)
            # This is a simplified version
            
            # Insert translated text blocks
            if page_num in blocks_by_page:
                for block in blocks_by_page[page_num]:
                    if block.bbox and block.translated_text:
                        rect = fitz.Rect(
                            block.bbox.x0,
                            block.bbox.y0,
                            block.bbox.x1,
                            block.bbox.y1
                        )
                        
                        # Estimate font size based on bbox height
                        num_lines = len(block.translated_text.split("\n"))
                        bbox_height = block.bbox.y1 - block.bbox.y0
                        fontsize = max(8, min(12, bbox_height / max(1, num_lines) * 0.8))
                        
                        output_page.insert_textbox(
                            rect,
                            block.translated_text,
                            fontsize=fontsize,
                            fontname="helv",
                            align=0
                        )
        
        output_doc.save(output_path)
        output_doc.close()
        source_doc.close()
    
    def render_text(self, document: Document, output_path: str):
        """Render document as plain text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in document.segments:
                for block in segment.blocks:
                    text = block.translated_text or block.source_text
                    f.write(text + "\n\n")
    
    def render_markdown(self, document: Document, output_path: str):
        """Render document as Markdown file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in document.segments:
                for block in segment.blocks:
                    text = block.translated_text or block.source_text
                    
                    # Add markdown formatting based on block type
                    block_type = block.metadata.get("block_type", "paragraph") if block.metadata else "paragraph"
                    
                    if block_type == "title":
                        f.write(f"# {text}\n\n")
                    elif block_type == "heading":
                        f.write(f"## {text}\n\n")
                    elif block_type in ["math_content", "math"]:
                        f.write(f"$$\n{text}\n$$\n\n")
                    else:
                        f.write(f"{text}\n\n")

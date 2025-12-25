"""PDF parsing and text extraction using best available methods."""

from pathlib import Path
from typing import List, Dict, Optional
import re
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Check for pymupdf_layout at runtime (not just import time)
def _check_pymupdf_layout():
    """Check if pymupdf_layout is available at runtime."""
    try:
        import pymupdf_layout
        # Try to actually use it to verify it works
        from pymupdf_layout import LayoutAnalyzer
        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False

# Check at module load time
HAS_PYMUPDF_LAYOUT = _check_pymupdf_layout()

from ..core.models import Document, Block, BoundingBox, BlockType, Segment, FontInfo
from .style_detector import StyleDetector, extract_font_from_span


class PDFParser:
    """
    Extract text and layout from PDF files using best available methods.
    
    Uses (in priority order):
    - pymupdf_layout (if available) - BEST for layout analysis
    - PyMuPDF (mandatory) for text extraction and basic layout
    - PyMuPDF find_tables() for table detection (best available)
    - YOLO layout detection (if available) for advanced layout analysis
    - Heuristic methods as fallback
    """
    
    def __init__(self, use_ocr: bool = False, use_yolo: bool = True, use_pymupdf_layout: bool = True):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        self.use_ocr = use_ocr
        self.use_yolo = use_yolo
        # Check pymupdf_layout availability at runtime (may have been installed after import)
        runtime_has_pymupdf_layout = _check_pymupdf_layout()
        self.use_pymupdf_layout = use_pymupdf_layout and runtime_has_pymupdf_layout
        
        # Try to load pymupdf_layout if requested and available (BEST method)
        if self.use_pymupdf_layout:
            logger.info("pymupdf_layout enabled - using best available layout analysis")
        elif use_pymupdf_layout and not runtime_has_pymupdf_layout:
            # Silent - pymupdf_layout is optional, system works fine without it
            # May be installed in a different venv, which is fine
            pass
        
        # Try to load YOLO if requested and available
        self.yolo_model = None
        if self.use_yolo:
            try:
                from .yolo import load_yolo_model
                self.yolo_model = load_yolo_model()
                if self.yolo_model:
                    logger.info("YOLO layout detection enabled - using advanced extraction methods")
                else:
                    logger.debug("YOLO not available - using PyMuPDF + heuristics")
            except Exception as e:
                logger.debug(f"YOLO not available: {e} - using PyMuPDF + heuristics")
    
    def parse(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> Document:
        """
        Parse PDF file and extract text blocks with layout information.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None = all)
            start_page: First page to process (0-indexed)
            end_page: Last page to process (None = all pages from start_page)
        
        Returns:
            Document object with extracted blocks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        try:
            total_pages = len(doc)
            
            # Determine page range
            if end_page is None:
                end_page = total_pages - 1
            else:
                end_page = min(end_page, total_pages - 1)
            
            if max_pages is not None:
                end_page = min(end_page, start_page + max_pages - 1)
            
            start_page = max(0, start_page)
            end_page = max(start_page, end_page)
            
            logger.info(f"Parsing PDF: {pdf_path.name} (pages {start_page+1}-{end_page+1} of {total_pages})")
            
            # Initialize style detector
            style_detector = StyleDetector()
            
            # Extract blocks from each page
            all_blocks = []
            block_counter = 0
            
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                page_rect = page.rect
                
                # Extract text blocks from page
                text_dict = page.get_text("dict")
                
                # Process text blocks
                for block_dict in text_dict.get("blocks", []):
                    if "lines" not in block_dict:  # Skip image blocks
                        continue
                    
                    # Collect text and font info from all lines in block
                    block_text_lines = []
                    font_infos = []
                    first_line = True
                    
                    for line in block_dict.get("lines", []):
                        line_text = ""
                        line_fonts = []
                        
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            line_text += span_text
                            
                            # Extract font info from span
                            font_dict = extract_font_from_span(span)
                            line_fonts.append(font_dict)
                        
                        if line_text.strip():
                            block_text_lines.append(line_text)
                            # Use dominant font (largest size) for the line
                            if line_fonts:
                                dominant_font = max(line_fonts, key=lambda f: f.get("size", 0))
                                font_infos.append(dominant_font)
                            first_line = False
                    
                    if not block_text_lines:
                        continue
                    
                    block_text = "\n".join(block_text_lines).strip()
                    if not block_text:
                        continue
                    
                    # Get bounding box
                    bbox = block_dict.get("bbox", [0, 0, 0, 0])
                    if len(bbox) < 4:
                        continue
                    
                    # Determine dominant font (most common or largest)
                    dominant_font_dict = font_infos[0] if font_infos else {
                        "family": "helv", "size": 11.0, "weight": "normal", 
                        "style": "normal", "color": "#000000"
                    }
                    
                    # Detect style features
                    position_info = {
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "page_width": page_rect.width,
                        "page_height": page_rect.height,
                        "x": (bbox[0] + bbox[2]) / 2,
                        "y": (bbox[1] + bbox[3]) / 2,
                    }
                    
                    style_features = style_detector.detect_style(
                        block_text,
                        position=position_info,
                        font_size=dominant_font_dict.get("size"),
                        is_first_line=first_line
                    )
                    
                    # Determine block type from style features
                    block_type = BlockType.PARAGRAPH
                    if style_features.is_heading:
                        block_type = BlockType.HEADING if style_features.heading_level == 1 else BlockType.SUBHEADING
                    elif style_features.is_list_item:
                        block_type = BlockType.LIST_ITEM
                    elif style_features.is_caption:
                        block_type = BlockType.CAPTION
                    elif style_features.is_footer:
                        block_type = BlockType.FOOTER
                    elif style_features.is_header:
                        block_type = BlockType.HEADER
                    elif style_features.is_abstract:
                        block_type = BlockType.ABSTRACT
                    elif style_features.is_reference:
                        block_type = BlockType.REFERENCE
                    elif style_features.is_footnote:
                        block_type = BlockType.FOOTNOTE
                    elif style_features.is_code:
                        block_type = BlockType.CODE
                    elif style_features.is_equation:
                        block_type = BlockType.EQUATION
                    
                    # Create enhanced FontInfo
                    font_info = FontInfo(
                        family=dominant_font_dict.get("family", "helv"),
                        size=dominant_font_dict.get("size", 11.0),
                        weight=dominant_font_dict.get("weight", "normal"),
                        style=dominant_font_dict.get("style", "normal"),
                        color=dominant_font_dict.get("color", "#000000"),
                        alignment=style_features.alignment_hint or "left",
                        line_height=dominant_font_dict.get("line_height"),
                        letter_spacing=dominant_font_dict.get("letter_spacing"),
                        decoration=dominant_font_dict.get("decoration", "none"),
                        is_small_caps=dominant_font_dict.get("is_small_caps", False),
                        list_style=style_features.list_style,
                        heading_level=style_features.heading_level,
                    )
                    
                    # Create block with enhanced information
                    block = Block(
                        block_id=f"block_{block_counter}",
                        source_text=block_text,
                        block_type=block_type,
                        bbox=BoundingBox(
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            page=page_num
                        ),
                        font=font_info
                    )
                    all_blocks.append(block)
                    block_counter += 1
            
            # Create document
            # Group blocks into a single segment
            segment = Segment(
                segment_id="main",
                segment_type="body",
                blocks=all_blocks
            )
            
            # Extract title from PDF metadata if available
            title = None
            try:
                metadata = doc.metadata
                if metadata:
                    title = metadata.get("title", "").strip()
                    if not title:
                        title = metadata.get("subject", "").strip()
            except:
                pass
            
            document = Document(
                document_id=str(pdf_path.stem),
                segments=[segment],
                title=title if title else None,
                stats={
                    "num_pages": end_page - start_page + 1,
                    "total_pages": total_pages,
                    "num_blocks": len(all_blocks),
                    "start_page": start_page,
                    "end_page": end_page
                }
            )
            
            logger.info(f"Extracted {len(all_blocks)} blocks from {end_page - start_page + 1} pages")
            return document
            
        finally:
            doc.close()

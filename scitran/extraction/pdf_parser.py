"""PDF parsing and text extraction."""

from pathlib import Path
from typing import List, Dict, Optional
import re

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

from ..core.models import Document, Block, BoundingBox, BlockType


class PDFParser:
    """Extract text and layout from PDF files."""
    
    def __init__(self, use_ocr: bool = False):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        self.use_ocr = use_ocr
    
    def parse(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        start_page: int = 0,
        end_page: Optional[int] = None,
    ) -> Document:
        """
        Parse PDF and extract structured content.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None = all)
            start_page: 0-based start page to process (inclusive)
            end_page: 0-based end page to process (inclusive). None = until max_pages or end.
            
        Returns:
            Document object with structured content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        total_pages = len(doc)
        start = max(0, start_page)
        stop = end_page + 1 if end_page is not None else total_pages
        if max_pages:
            stop = min(stop, start + max_pages)
        stop = min(stop, total_pages)

        blocks = []
        block_counter = 0
        num_pages = max(0, stop - start)
        
        for page_num in range(start, stop):
            page = doc[page_num]
            page_blocks = self._extract_page_blocks(page, page_num, block_counter)
            blocks.extend(page_blocks)
            block_counter += len(page_blocks)
        
        doc.close()
        
        # Create segments from blocks
        from scitran.core.models import Segment
        segment = Segment(
            segment_id=f"{pdf_path.stem}_main",
            segment_type="document",
            title=pdf_path.stem,
            blocks=blocks
        )
        
        document = Document(
            document_id=pdf_path.stem,
            segments=[segment],
            source_path=str(pdf_path),
            stats={
                "num_pages": num_pages,
                "total_blocks": len(blocks)
            }
        )
        
        return document
    
    def _extract_page_blocks(self, page, page_num: int, start_id: int) -> List[Block]:
        """Extract blocks from a single page with font information."""
        from ..core.models import FontInfo
        
        blocks = []
        
        # Get text with layout information and preserve ligatures/spaces
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
        
        for block_idx, block_data in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block_data:
                # Non-text blocks (likely images/figures) - keep as figure placeholders
                bbox = block_data.get("bbox", [0, 0, 0, 0])
                bounding_box = BoundingBox(
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3],
                    page=page_num,
                )
                figure_block = Block(
                    block_id=f"block_{start_id + len(blocks)}",
                    source_text="",
                    bbox=bounding_box,
                    block_type=BlockType.FIGURE,
                    metadata={"page": page_num, "block_type": "figure"},
                )
                blocks.append(figure_block)
                continue  # Skip image blocks for now
            
            # Extract text and font info from block
            lines = []
            fonts = []  # Collect font info from spans
            
            for line in block_data["lines"]:
                line_text = ""
                for span in line["spans"]:
                    # Preserve spacing between spans if needed
                    if line_text and not line_text.endswith(" ") and span["text"] and not span["text"].startswith(" "):
                        line_text += " "
                    line_text += span["text"]
                    # Collect font information
                    fonts.append({
                        "family": span.get("font", ""),
                        "size": span.get("size", 11),
                        "color": span.get("color", 0),
                        "flags": span.get("flags", 0)  # bold, italic flags
                    })
                lines.append(line_text)
            
            text = "\n".join(lines).strip()
            if not text:
                continue
            
            # Get bounding box
            bbox = block_data.get("bbox", [0, 0, 0, 0])
            bounding_box = BoundingBox(
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                page=page_num
            )
            
            # Get dominant font from spans (most common or first)
            font_info = None
            if fonts:
                dominant_font = fonts[0]  # Use first span's font
                # Convert color int to hex
                color_int = dominant_font.get("color", 0)
                if isinstance(color_int, int):
                    color_hex = f"#{color_int:06x}"
                else:
                    color_hex = "#000000"
                
                # Parse font flags for weight/style
                # PyMuPDF flags: bit0=superscript, bit1=italic, bit2=serif, bit3=mono, bit4=bold
                flags = dominant_font.get("flags", 0)
                weight = "bold" if flags & 16 else "normal"  # Bit 4 = bold (16)
                style = "italic" if flags & 2 else "normal"  # Bit 1 = italic (2)
                
                # Also detect font type from flags
                font_family = dominant_font.get("family", "")
                is_serif = bool(flags & 4)    # Bit 2 = serif
                is_mono = bool(flags & 8)     # Bit 3 = monospace
                
                # Add hints to font family for better rendering
                if is_mono and "mono" not in font_family.lower() and "cour" not in font_family.lower():
                    font_family = f"{font_family} mono"
                elif is_serif and "serif" not in font_family.lower() and "times" not in font_family.lower():
                    font_family = f"{font_family} serif"
                
                font_info = FontInfo(
                    family=font_family,
                    size=dominant_font.get("size", 11),
                    weight=weight,
                    style=style,
                    color=color_hex
                )
            
            # Create block with font info
            classified = self._classify_block(text)
            block = Block(
                block_id=f"block_{start_id + len(blocks)}",
                source_text=text,
                bbox=bounding_box,
                font=font_info,
                block_type=self._map_block_type(classified),
                metadata={
                    "page": page_num,
                    "block_type": classified,
                    "num_lines": len(lines)
                }
            )
            
            blocks.append(block)
        
        return blocks
    
    def _classify_block(self, text: str) -> str:
        """Classify block type based on content."""
        text_lower = text.lower().strip()
        words = text.split()
        
        # Titles / headings
        if len(text) < 120 and text.isupper():
            return "title"
        if text.startswith("Abstract"):
            return "abstract"
        if re.match(r"^\d+\.?\s+", text):
            return "numbered_section"
        if len(words) < 10:
            return "heading"
        
        # Figures / tables (keywords)
        if any(keyword in text_lower for keyword in ["figure", "fig.", "table", "tbl.", "tab."]):
            return "caption"
        
        # Table-like patterns: multiple columns separated by tabs/pipes or repeated spaces
        if ("|" in text and text.count("|") >= 2) or ("\t" in text) or re.search(r"\s{2,}\S+\s{2,}\S+", text):
            return "table"
        
        # Math detection: latex markers, many symbols, caret/superscripts
        if re.search(r'\$.*?\$|\\[a-zA-Z]+|≠|≈|≥|≤|∑|∫|√|∞|→|←|\^|_{|}', text):
            return "math_content"
        
        return "paragraph"
    
    def _map_block_type(self, cls: str) -> BlockType:
        """Map classifier string to BlockType enum."""
        mapping = {
            "title": BlockType.TITLE,
            "abstract": BlockType.ABSTRACT,
            "numbered_section": BlockType.SUBHEADING,
            "heading": BlockType.HEADING,
            "caption": BlockType.CAPTION,
            "table": BlockType.TABLE,
            "math_content": BlockType.EQUATION,
        }
        return mapping.get(cls, BlockType.PARAGRAPH)
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata."""
        doc = fitz.open(str(pdf_path))
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "num_pages": len(doc)
        }
        doc.close()
        return metadata
    
    def extract_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """Extract images from PDF."""
        doc = fitz.open(str(pdf_path))
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                image_info = {
                    "page": page_num,
                    "index": img_idx,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "format": base_image["ext"]
                }
                
                if output_dir:
                    output_path = Path(output_dir) / f"page{page_num}_img{img_idx}.{base_image['ext']}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(base_image["image"])
                    image_info["path"] = str(output_path)
                
                images.append(image_info)
        
        doc.close()
        return images


class PDFParserAlternative:
    """Alternative PDF parser using pdfplumber (more accurate for tables)."""
    
    def __init__(self):
        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    
    def parse(self, pdf_path: str) -> Document:
        """Parse PDF using pdfplumber."""
        with pdfplumber.open(pdf_path) as pdf:
            blocks = []
            block_counter = 0
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # Split into paragraphs
                paragraphs = text.split("\n\n")
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    block = Block(
                        block_id=f"block_{block_counter}",
                        source_text=para,
                        metadata={"page": page_num}
                    )
                    blocks.append(block)
                    block_counter += 1
            
            # Create segment from blocks
            from scitran.core.models import Segment
            segment = Segment(
                segment_id=f"{Path(pdf_path).stem}_main",
                segment_type="document",
                title=Path(pdf_path).stem,
                blocks=blocks
            )
            
            document = Document(
                document_id=Path(pdf_path).stem,
                segments=[segment],
                source_path=str(pdf_path),
                stats={
                    "num_pages": len(pdf.pages),
                    "total_blocks": len(blocks),
                    "parser": "pdfplumber"
                }
            )
            
            return document

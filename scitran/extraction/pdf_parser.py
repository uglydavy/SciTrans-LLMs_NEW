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

from ..core.models import Document, Block, BoundingBox


class PDFParser:
    """Extract text and layout from PDF files."""
    
    def __init__(self, use_ocr: bool = False):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        self.use_ocr = use_ocr
    
    def parse(self, pdf_path: str, max_pages: Optional[int] = None) -> Document:
        """
        Parse PDF and extract structured content.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None = all)
            
        Returns:
            Document object with structured content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        blocks = []
        block_counter = 0
        
        num_pages = min(len(doc), max_pages) if max_pages else len(doc)
        
        for page_num in range(num_pages):
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
        """Extract blocks from a single page."""
        blocks = []
        
        # Get text with layout information
        text_dict = page.get_text("dict")
        
        for block_idx, block_data in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block_data:
                continue  # Skip image blocks for now
            
            # Extract text from block
            lines = []
            for line in block_data["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
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
            
            # Create block
            block = Block(
                block_id=f"block_{start_id + len(blocks)}",
                source_text=text,
                bbox=bounding_box,
                metadata={
                    "page": page_num,
                    "block_type": self._classify_block(text),
                    "num_lines": len(lines)
                }
            )
            
            blocks.append(block)
        
        return blocks
    
    def _classify_block(self, text: str) -> str:
        """Classify block type based on content."""
        text_lower = text.lower().strip()
        
        # Check for common patterns
        if len(text) < 100 and text.isupper():
            return "title"
        
        if re.match(r"^\d+\.?\s+", text):
            return "numbered_section"
        
        if text.startswith("Abstract"):
            return "abstract"
        
        if any(keyword in text_lower for keyword in ["figure", "fig.", "table"]):
            return "caption"
        
        if re.search(r'\$.*\$|\\[a-z]+', text):
            return "math_content"
        
        if len(text.split()) < 10:
            return "heading"
        
        return "paragraph"
    
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

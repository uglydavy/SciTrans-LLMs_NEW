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
    
    def _detect_protected_zones(self, page) -> Dict[str, List]:
        """Detect protected zones (tables, figures, images) on a page.
        
        Returns:
            Dict with 'table_zones', 'image_zones', 'drawing_zones' as lists of fitz.Rect
        """
        table_zones = []
        image_zones = []
        drawing_zones = []
        
        # 1. TABLE DETECTION using PyMuPDF's find_tables()
        try:
            tables = page.find_tables()
            if tables:
                for table in tables.tables:
                    if hasattr(table, 'bbox'):
                        table_rect = fitz.Rect(table.bbox)
                        if not table_rect.is_empty and table_rect.get_area() > 100:
                            table_zones.append(table_rect)
        except Exception as e:
            # find_tables might not be available in all PyMuPDF versions
            pass
        
        # 2. IMAGE DETECTION
        try:
            for img in page.get_images():
                img_rect = page.get_image_bbox(img)
                if img_rect and not img_rect.is_empty:
                    image_zones.append(img_rect)
        except:
            pass
        
        # 3. VECTOR GRAPHICS (DRAWINGS) DETECTION - CRITICAL for scientific papers
        try:
            drawings = page.get_drawings()
            if drawings:
                # Collect all drawing rects
                drawing_rects = []
                for drawing in drawings:
                    if 'rect' in drawing:
                        rect = drawing['rect']
                        # Filter out tiny/trivial rects (lines, underlines, etc.)
                        if rect.width > 2 and rect.height > 2 and rect.get_area() > 10:
                            drawing_rects.append(rect)
                
                # Cluster overlapping/nearby rects into figures
                page_rect = page.rect
                page_area = page_rect.get_area()
                
                if drawing_rects:
                    clusters = self._cluster_rects(drawing_rects, distance_threshold=10)
                    
                    for cluster in clusters:
                        # Merge cluster into single bbox
                        merged = cluster[0]
                        for rect in cluster[1:]:
                            merged = merged | rect  # Union
                        
                        # Keep cluster only if it looks like a real figure
                        cluster_area = merged.get_area()
                        area_ratio = cluster_area / page_area if page_area > 0 else 0
                        
                        # Criteria: >= 2% of page area AND >= 10 drawing elements
                        if area_ratio >= 0.02 and len(cluster) >= 10:
                            drawing_zones.append(merged)
        except Exception as e:
            # get_drawings might fail on some PDFs
            pass
        
        return {
            'table_zones': table_zones,
            'image_zones': image_zones,
            'drawing_zones': drawing_zones
        }
    
    def _cluster_rects(self, rects: List, distance_threshold: float = 10) -> List[List]:
        """Cluster rects that are close or overlapping into groups.
        
        Args:
            rects: List of fitz.Rect objects
            distance_threshold: Maximum distance between rects in same cluster
        
        Returns:
            List of clusters, where each cluster is a list of rects
        """
        if not rects:
            return []
        
        import fitz
        clusters = [[rects[0]]]
        
        for rect in rects[1:]:
            added = False
            for cluster in clusters:
                # Check if rect is close to any rect in cluster
                for cluster_rect in cluster:
                    # Check overlap
                    overlap = rect.intersect(cluster_rect)
                    if overlap and not overlap.is_empty:
                        cluster.append(rect)
                        added = True
                        break
                    
                    # Check distance
                    # Simple distance: min distance between edges
                    dx = max(0, max(cluster_rect.x0 - rect.x1, rect.x0 - cluster_rect.x1))
                    dy = max(0, max(cluster_rect.y0 - rect.y1, rect.y0 - cluster_rect.y1))
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance <= distance_threshold:
                        cluster.append(rect)
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                clusters.append([rect])
        
        return clusters
    
    def _compute_ink_bbox(self, block_data: Dict, page) -> Optional[fitz.Rect]:
        """Compute tight bbox from non-whitespace span bboxes.
        
        Args:
            block_data: Block dict from get_text("dict")
            page: PyMuPDF page object
        
        Returns:
            Tight ink bbox or None
        """
        import fitz
        ink_rects = []
        
        for line in block_data.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text and text.strip():  # Non-whitespace
                    span_bbox = span.get("bbox")
                    if span_bbox:
                        ink_rects.append(fitz.Rect(span_bbox))
        
        if not ink_rects:
            return None
        
        # Union all span rects
        ink_bbox = ink_rects[0]
        for rect in ink_rects[1:]:
            ink_bbox = ink_bbox | rect
        
        return ink_bbox
    
    def _extract_page_blocks(self, page, page_num: int, start_id: int) -> List[Block]:
        """Extract blocks from a single page with protected zones detection."""
        from ..core.models import FontInfo
        import fitz
        
        blocks = []
        
        # STEP 1: Detect protected zones (tables, images, vector figures)
        protected_zones = self._detect_protected_zones(page)
        table_zones = protected_zones['table_zones']
        image_zones = protected_zones['image_zones']
        drawing_zones = protected_zones['drawing_zones']
        
        # Combine all protected zones
        all_protected = table_zones + image_zones + drawing_zones
        
        # STEP 2: Extract text with layout information
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
        
        for block_idx, block_data in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block_data:
                # Non-text blocks (images/figures without text) - skip
                # We already have them in protected zones
                continue
            
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
            
            # STEP 2.1: Compute INK BBOX (tight, span-based)
            ink_bbox = self._compute_ink_bbox(block_data, page)
            
            # Fallback to block bbox if ink bbox computation fails
            raw_bbox = block_data.get("bbox", [0, 0, 0, 0])
            if ink_bbox and not ink_bbox.is_empty:
                bbox = [ink_bbox.x0, ink_bbox.y0, ink_bbox.x1, ink_bbox.y1]
            else:
                bbox = raw_bbox
            
            bounding_box = BoundingBox(
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                page=page_num
            )
            
            block_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # STEP 2.2: ZONE-BASED CLASSIFICATION (CRITICAL for preservation)
            protected_reason = None
            zone_based_type = None
            
            # Check overlap with protected zones
            for table_rect in table_zones:
                overlap = block_rect.intersect(table_rect)
                if overlap and not overlap.is_empty:
                    overlap_ratio = overlap.get_area() / block_rect.get_area() if block_rect.get_area() > 0 else 0
                    if overlap_ratio > 0.20:  # 20% overlap threshold
                        zone_based_type = BlockType.TABLE
                        protected_reason = "inside_table_zone"
                        break
            
            if not zone_based_type:
                # Check images
                for img_rect in image_zones:
                    overlap = block_rect.intersect(img_rect)
                    if overlap and not overlap.is_empty:
                        overlap_ratio = overlap.get_area() / block_rect.get_area() if block_rect.get_area() > 0 else 0
                        if overlap_ratio > 0.30:
                            zone_based_type = BlockType.FIGURE
                            protected_reason = "inside_image_zone"
                            break
            
            if not zone_based_type:
                # Check vector graphics (drawings)
                for drawing_rect in drawing_zones:
                    overlap = block_rect.intersect(drawing_rect)
                    if overlap and not overlap.is_empty:
                        overlap_ratio = overlap.get_area() / block_rect.get_area() if block_rect.get_area() > 0 else 0
                        if overlap_ratio > 0.25:
                            zone_based_type = BlockType.FIGURE
                            protected_reason = "inside_drawing_zone"
                            break
            
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
            
            # STEP 2.3: Caption detection (near protected zones but NOT inside)
            is_caption = False
            if not zone_based_type:
                # Check if this looks like a caption
                caption_keywords = ['figure', 'fig.', 'table', 'tbl.', 'tab.', 'equation', 'eq.']
                text_lower_start = text.lower()[:50]
                
                if any(kw in text_lower_start for kw in caption_keywords):
                    # Check proximity to protected zones
                    for protected_rect in all_protected:
                        # Check vertical distance
                        if block_rect.y0 > protected_rect.y1:
                            # Below protected zone
                            distance = block_rect.y0 - protected_rect.y1
                        elif block_rect.y1 < protected_rect.y0:
                            # Above protected zone
                            distance = protected_rect.y0 - block_rect.y1
                        else:
                            distance = 0
                        
                        # Caption if within 60px and not overlapping significantly
                        overlap = block_rect.intersect(protected_rect)
                        overlap_ratio = overlap.get_area() / block_rect.get_area() if block_rect.get_area() > 0 and overlap else 0
                        
                        if distance <= 60 and overlap_ratio < 0.10:
                            is_caption = True
                            break
            
            # STEP 2.4: Final classification
            if zone_based_type:
                # Zone-based override (TABLE or FIGURE)
                final_type = zone_based_type
                classified = zone_based_type.name.lower()
            elif is_caption:
                final_type = BlockType.CAPTION
                classified = "caption"
            else:
                # Use content-based classification
                classified = self._classify_block(text)
                final_type = self._map_block_type(classified)
            
            # Create block with font info
            block = Block(
                block_id=f"block_{start_id + len(blocks)}",
                source_text=text,
                bbox=bounding_box,
                font=font_info,
                block_type=final_type,
                metadata={
                    "page": page_num,
                    "block_type": classified,
                    "num_lines": len(lines),
                    "protected_reason": protected_reason,  # Set if inside protected zone
                    "raw_bbox": raw_bbox if ink_bbox else None,  # Store original bbox
                    "is_caption": is_caption
                }
            )
            
            blocks.append(block)
        
        return blocks
    
    def _classify_block(self, text: str) -> str:
        """Classify block type based on content with improved table detection."""
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
        
        # Improved table detection with stricter criteria
        # Strong indicators: pipes or tabs
        if ("|" in text and text.count("|") >= 2) or ("\t" in text and text.count("\t") >= 2):
            return "table"
        
        # Check for table-like structure (multiple lines with aligned columns)
        lines = text.split('\n')
        if len(lines) >= 3:
            # Count lines with multiple wide spaces (potential columns)
            space_patterns = [len(re.findall(r'\s{3,}', line)) for line in lines[:5]]
            lines_with_columns = len([p for p in space_patterns if p >= 2])
            
            # Also check digit ratio (tables often have many numbers)
            digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
            
            # Only classify as table if:
            # - Multiple lines have column-like spacing AND
            # - High digit ratio (>20%) OR multiple lines have similar patterns
            if lines_with_columns >= 2:
                if digit_ratio > 0.2 or lines_with_columns >= 3:
                    return "table"
            
            # Check for tab-separated values (strong indicator)
            if sum(1 for line in lines[:5] if '\t' in line) >= 3:
                return "table"
        
        # Enhanced math detection: latex markers, many symbols, caret/superscripts
        # Also detect standalone equations (lines that are mostly math)
        math_patterns = [
            r'\$.*?\$',  # LaTeX inline math
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'≠|≈|≥|≤|∑|∫|√|∞|→|←|×|÷|±|∓',  # Math symbols
            r'\^[0-9]|_[0-9]',  # Superscripts/subscripts
            r'\\begin\{equation\}|\\begin\{align\}|\\begin\{eqnarray\}',  # LaTeX equation environments
            r'\\frac\{.*?\}\{.*?\}',  # Fractions
        ]
        for pattern in math_patterns:
            if re.search(pattern, text):
                return "math_content"
        
        # Check if line is mostly math (high ratio of math symbols to text)
        if len(text) > 0:
            math_chars = len(re.findall(r'[=+\-*/^_()\[\]{}|\\$]', text))
            alpha_chars = len(re.findall(r'[a-zA-Z]', text))
            if math_chars > 0 and (math_chars / max(len(text), 1)) > 0.3:
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

# -*- coding: utf-8 -*-
"""PDF rendering with proper layout preservation - clears source text before placing translated."""

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import logging
import tempfile
import os

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from ..core.models import Document, Block, FontInfo, BlockType

logger = logging.getLogger(__name__)

# Font mapping from common PDF fonts to PyMuPDF base14 fonts
FONT_MAP = {
    # Times family
    "times": "times-roman",
    "times new roman": "times-roman",
    "timesnewroman": "times-roman",
    "timesnewromanps": "times-roman",
    "timesroman": "times-roman",
    # Helvetica/Arial family  
    "arial": "helv",
    "helvetica": "helv",
    "arialmt": "helv",
    "helveticaneue": "helv",
    # Courier family
    "courier": "cour",
    "courier new": "cour",
    "couriernew": "cour",
    # Symbol
    "symbol": "symb",
    "zapfdingbats": "zadb",
    # CMR (Computer Modern - LaTeX)
    "cmr": "times-roman",
    "cmbx": "times-bold",
    "cmti": "times-italic",
    "cmsy": "symb",
}


class PDFRenderer:
    """Render translated documents to PDF with proper layout preservation."""
    
    def __init__(self, strict_mode: bool = False, embed_fonts: bool = False):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        self.default_font = "helv"
        self.default_fontsize = 11
        self.temp_files = []  # Track temp files for cleanup
        self.strict_mode = strict_mode
        self.embed_fonts = embed_fonts
    
    def cleanup(self):
        """Clean up temporary files."""
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        self.temp_files = []
    
    def render_pdf(self, document: Document, output_path: str, source_pdf: str = None):
        """Main entry point for PDF rendering."""
        if source_pdf:
            return self.render_with_layout(source_pdf, document, output_path)
        return self.render_simple(document, output_path)
    
    def _get_font_name(self, font_info: Optional[FontInfo]) -> Tuple[str, bool, bool]:
        """
        Map font info to available PDF font with better style preservation.
        
        Returns:
            (fontname, is_bold, is_italic)
        """
        if not font_info:
            return self.default_font, False, False
        
        family = font_info.family.lower().replace(" ", "").replace("-", "") if font_info.family else ""
        
        # Detect bold from weight or font name
        is_bold = False
        if font_info.weight:
            is_bold = font_info.weight.lower() in ["bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold"]
        if not is_bold and any(b in family for b in ["bold", "black", "heavy", "semibold", "demi"]):
            is_bold = True
        
        # Detect italic from style or font name
        is_italic = False
        if font_info.style:
            is_italic = font_info.style.lower() in ["italic", "oblique", "slanted"]
        if not is_italic and any(i in family for i in ["italic", "oblique", "slant", "incline"]):
            is_italic = True
        
        # Determine if serif or sans-serif based on font family name
        is_serif = False
        serif_indicators = ["times", "roman", "serif", "georgia", "palatino", "garamond", "cambria", "charter", "bookman", "cm"]
        sans_indicators = ["arial", "helvetica", "helv", "sans", "verdana", "tahoma", "calibri", "segoe", "roboto", "opensans", "lato"]
        mono_indicators = ["courier", "mono", "consolas", "menlo", "code", "fixed", "terminal"]
        
        for indicator in serif_indicators:
            if indicator in family:
                is_serif = True
                break
        
        is_mono = any(m in family for m in mono_indicators)
        
        # Select base font based on classification
        if is_mono:
            base_font = "cour"
        elif is_serif:
            base_font = "times-roman"
        else:
            # Default to sans-serif (more common in modern documents)
            base_font = "helv"
        
        # Also check the explicit font map for known fonts
        for key, value in FONT_MAP.items():
            if key in family:
                base_font = value
                break
        
        # Apply bold/italic variants using PyMuPDF Base14 font names
        # See: https://pymupdf.readthedocs.io/en/latest/app1.html
        if base_font in ["helv", "helvetica"]:
            if is_bold and is_italic:
                return "hebo", is_bold, is_italic  # Helvetica-BoldOblique
            elif is_bold:
                return "hebo", is_bold, is_italic  # Helvetica-Bold (use hebo for both)
            elif is_italic:
                return "heob", is_bold, is_italic  # Helvetica-Oblique
            return "helv", is_bold, is_italic
        elif base_font in ["times-roman", "tiro", "times"]:
            if is_bold and is_italic:
                return "tibi", is_bold, is_italic  # Times-BoldItalic
            elif is_bold:
                return "tibo", is_bold, is_italic  # Times-Bold
            elif is_italic:
                return "tiit", is_bold, is_italic  # Times-Italic
            return "tiro", is_bold, is_italic  # Times-Roman
        elif base_font in ["cour", "courier"]:
            if is_bold and is_italic:
                return "cobo", is_bold, is_italic  # Courier-BoldOblique
            elif is_bold:
                return "cobo", is_bold, is_italic  # Courier-Bold
            elif is_italic:
                return "cooo", is_bold, is_italic  # Courier-Oblique
            return "cour", is_bold, is_italic
        
        # For unknown fonts, return with style flags
        return base_font, is_bold, is_italic
    
    def render_simple(self, document: Document, output_path: str):
        """Simple rendering without source PDF."""
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
            page = pdf.new_page(width=595, height=842)  # A4
            y_position = 50
            
            for block in blocks_by_page[page_num]:
                text = block.translated_text or block.source_text
                rect = fitz.Rect(50, y_position, 545, y_position + 200)
                page.insert_textbox(rect, text, fontsize=11, fontname="helv", align=0)
                y_position += len(text.split("\n")) * 15 + 10
                if y_position > 750:
                    break
        
        pdf.save(output_path)
        pdf.close()
    
    def _extract_page_layout(self, page) -> Dict[str, Any]:
        """Extract complete layout information from a page."""
        layout = {
            "width": page.rect.width,
            "height": page.rect.height,
            "text_blocks": [],
            "images": [],
            "drawings": [],
        }
        
        # Extract text blocks with detailed info
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        layout["text_blocks"].append({
                            "bbox": span.get("bbox"),
                            "text": span.get("text", ""),
                            "font": span.get("font", ""),
                            "size": span.get("size", 11),
                            "color": span.get("color", 0),
                            "flags": span.get("flags", 0),
                        })
        
        # Extract images
        try:
            for img in page.get_images():
                xref = img[0]
                layout["images"].append({
                    "xref": xref,
                    "bbox": page.get_image_bbox(img),
                })
        except:
            pass
        
        # Extract drawings
        try:
            layout["drawings"] = page.get_drawings()
        except:
            pass
        
        return layout
    
    def _redact_text_from_page(self, page, blocks: List[Block]) -> None:
        """Remove source text from specific regions using redaction."""
        redact_count = 0
        
        for block in blocks:
            rect = fitz.Rect(
                block.bbox.x0,
                block.bbox.y0,
                block.bbox.x1,
                block.bbox.y1
            )
            
            # Validate rect
            if rect.is_empty or rect.is_infinite:
                logger.warning(f"Skipping invalid rect for redaction: {rect}")
                continue
                
            # Add redaction annotation with white fill and no text
            page.add_redact_annot(
                rect, 
                fill=(1, 1, 1),  # White fill
                text="",  # No replacement text
            )
            redact_count += 1
        
        # Apply all redactions - this removes the content
        if redact_count > 0:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            logger.info(f"Applied {redact_count} redactions on page")
    
    def render_with_layout(self, source_pdf: str, document: Document, output_path: str):
        """
        Render document preserving layout by:
        1. Opening source PDF
        2. Redacting (removing) source text in translation regions
        3. Inserting translated text in same positions
        4. Preserving images and drawings
        """
        # Create a working copy of the source PDF
        temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = temp_pdf.name
        temp_pdf.close()
        self.temp_files.append(temp_path)
        
        # Copy source to temp
        source_doc = fitz.open(source_pdf)
        source_doc.save(temp_path)
        # Keep source_doc open for region stamping
        
        # Open working copy
        doc = fitz.open(temp_path)
        
        # Group blocks by page
        blocks_by_page: Dict[int, List[Block]] = {}
        translated_count = 0
        skipped_no_bbox = 0
        skipped_no_translation = 0
        preserve_by_page: Dict[int, List[Block]] = {}
        
        total_pages_in_doc = len(doc)
        logger.info(f"Source PDF has {total_pages_in_doc} pages")
        
        for segment in document.segments:
            for block in segment.blocks:
                if not block.bbox:
                    skipped_no_bbox += 1
                    continue
                
                page_num = block.bbox.page
                
                # Collect preservable blocks (tables, equations, figures, captions)
                if block.block_type in {
                    BlockType.TABLE,
                    BlockType.EQUATION,
                    BlockType.FIGURE,
                    BlockType.CAPTION,
                }:
                    preserve_by_page.setdefault(page_num, []).append(block)
                
                # Only translated blocks go to redaction/insertion
                if not block.translated_text:
                    skipped_no_translation += 1
                    continue
                    
                if page_num not in blocks_by_page:
                    blocks_by_page[page_num] = []
                blocks_by_page[page_num].append(block)
                translated_count += 1
        
        logger.info(f"Rendering {translated_count} translated blocks")
        logger.info(f"Skipped: {skipped_no_bbox} (no bbox), {skipped_no_translation} (no translation)")
        logger.info(f"Pages with translations: {sorted(blocks_by_page.keys())}")
        
        # Process each page
        pages_processed = 0
        for page_num in range(total_pages_in_doc):
            page = doc[page_num]
            
            if page_num in blocks_by_page:
                page_blocks = blocks_by_page[page_num]
                logger.info(f"Page {page_num + 1}/{total_pages_in_doc}: processing {len(page_blocks)} blocks")
                
                # Step 0: Preserve tables/equations/figures by stamping source region
                if page_num in preserve_by_page:
                    self._stamp_preserved_blocks(page, source_doc[page_num], preserve_by_page[page_num])
                
                # Step 1: Redact (remove) source text in translation regions
                self._redact_text_from_page(page, page_blocks)
                
                # Step 2: Insert translated text
                for block in page_blocks:
                    self._insert_text_block(page, block)
                pages_processed += 1
            else:
                # Still stamp preserved blocks even if no translations
                if page_num in preserve_by_page:
                    self._stamp_preserved_blocks(page, source_doc[page_num], preserve_by_page[page_num])
                logger.info(f"Page {page_num + 1}/{total_pages_in_doc}: no translated blocks")
        
        logger.info(f"Processed {pages_processed} pages with translations out of {total_pages_in_doc} total pages")
        
        # Save output (embed fonts if requested)
        save_opts = {"garbage": 4, "deflate": True}
        if self.embed_fonts:
            # Subset fonts to preserve styling as much as possible
            doc.subset_fonts()
        doc.save(output_path, **save_opts)
        doc.close()
        source_doc.close()
        
        # Cleanup temp
        self.cleanup()
        
        logger.info(f"PDF rendered to {output_path}")
    
    def _insert_text_block(self, page, block: Block):
        """Insert a translated text block with proper formatting preserving original style."""
        rect = fitz.Rect(
            block.bbox.x0,
            block.bbox.y0,
            block.bbox.x1,
            block.bbox.y1
        )
        
        # Validate rect
        if rect.is_empty or rect.is_infinite:
            logger.warning(f"Invalid rect for block {block.block_id}: {rect}")
            return
        
        # Get font info with bold/italic
        fontname, is_bold, is_italic = self._get_font_name(block.font)
        
        # Get font size from original block - PRESERVE EXACT SIZE
        if block.font and block.font.size and block.font.size > 0:
            fontsize = block.font.size
            logger.debug(f"Block {block.block_id}: Using extracted font size {fontsize}")
        else:
            # Estimate from bbox height - be more conservative
            bbox_height = rect.height
            num_lines = max(1, block.translated_text.count('\n') + 1)
            # More accurate estimation: typical line height is ~1.2x font size
            estimated_size = bbox_height / num_lines / 1.2
            fontsize = max(8, min(20, estimated_size))  # Allow larger fonts
            logger.debug(f"Block {block.block_id}: Estimated font size {fontsize} from bbox height {bbox_height}")
        
        # Get color (default black)
        color = (0, 0, 0)
        if block.font and block.font.color:
            try:
                hex_color = block.font.color.lstrip('#')
                if len(hex_color) == 6:
                    color = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
                    logger.debug(f"Block {block.block_id}: Using color {color} from {hex_color}")
            except:
                pass
        
        text = block.translated_text
        if not text or not text.strip():
            return
        
        # PRESERVE ORIGINAL FONT SIZE - only reduce if absolutely necessary
        # Try original size first, then minimal reductions
        min_size = 6 if self.strict_mode else 7
        sizes_to_try = [fontsize, fontsize * 0.95, fontsize * 0.9, fontsize * 0.85, max(min_size, fontsize * 0.75)]
        
        for size in sizes_to_try:
            try:
                # Use insert_textbox for multi-line text fitting
                # Preserve line spacing based on original font size
                line_height = 1.15 if fontsize > 12 else 1.1
                
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontsize=size,
                    fontname=fontname,
                    color=color,
                    align=0,  # Left align
                    lineheight=line_height
                )
                
                # rc >= 0 means text fit perfectly
                if rc >= 0:
                    logger.debug(f"✓ Inserted block {block.block_id}: font={fontname}, size={size:.1f}, bold={is_bold}, italic={is_italic}")
                    return
                elif size == sizes_to_try[-1]:
                    # Last attempt - accept even with overflow (text was inserted)
                    msg = f"Overflow accepted (strict={self.strict_mode}) block {block.block_id}: font={fontname}, size={size:.1f}"
                    if self.strict_mode:
                        logger.warning(msg)
                    else:
                        logger.debug(f"⚠ {msg}")
                    return
                    
            except Exception as e:
                logger.debug(f"Insert attempt failed for {block.block_id} at size {size}: {e}")
                continue
        
        # Final fallback - use original font size even if overflow
        try:
            page.insert_textbox(
                rect, 
                text, 
                fontsize=fontsize,  # Use original size
                fontname=fontname,  # Use original font
                color=color,  # Use original color
                align=0
            )
            logger.debug(f"Fallback insert for block {block.block_id}: font={fontname}, size={fontsize}")
        except Exception as e:
            logger.error(f"All insert attempts failed for block {block.block_id}: {e}")
    
    def _stamp_preserved_blocks(self, target_page, source_page, blocks: List[Block]):
        """Stamp source regions for tables/equations/figures to preserve appearance."""
        for block in blocks:
            if not block.bbox:
                continue
            rect = fitz.Rect(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
            if rect.is_empty or rect.is_infinite:
                continue
            try:
                # Clip region from source page
                pix = source_page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=rect)
                img_stream = pix.tobytes("png")
                target_page.insert_image(rect, stream=img_stream, keep_proportion=False, overlay=True)
                logger.debug(f"Stamped preserved region for block {block.block_id} ({block.block_type})")
            except Exception as e:
                logger.warning(f"Failed to stamp preserved block {block.block_id}: {e}")
    
    def render_clean_translation(self, source_pdf: str, document: Document, output_path: str):
        """
        Alternative approach: Create clean output by:
        1. Creating blank pages with same dimensions
        2. Copying all images and drawings from source
        3. Placing translated text at source text positions
        
        This produces cleaner output but loses any untranslated text.
        """
        source_doc = fitz.open(source_pdf)
        output_doc = fitz.open()
        
        # Group blocks by page
        blocks_by_page: Dict[int, List[Block]] = {}
        for segment in document.segments:
            for block in segment.blocks:
                page_num = block.bbox.page if block.bbox else 0
                if page_num not in blocks_by_page:
                    blocks_by_page[page_num] = []
                blocks_by_page[page_num].append(block)
        
        for page_num in range(len(source_doc)):
            source_page = source_doc[page_num]
            
            # Create blank page
            output_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height
            )
            
            # Copy images
            self._copy_images(source_doc, source_page, output_page)
            
            # Copy drawings
            self._copy_drawings(source_page, output_page)
            
            # Insert translated text
            if page_num in blocks_by_page:
                for block in blocks_by_page[page_num]:
                    if block.bbox and block.translated_text:
                        self._insert_text_block(output_page, block)
        
        output_doc.save(output_path)
        output_doc.close()
        source_doc.close()
    
    def _copy_images(self, source_doc, source_page, output_page):
        """Copy images from source page to output page."""
        try:
            for img_info in source_page.get_images():
                xref = img_info[0]
                try:
                    base_image = source_doc.extract_image(xref)
                    img_rect = source_page.get_image_bbox(img_info)
                    if img_rect and base_image.get("image"):
                        output_page.insert_image(img_rect, stream=base_image["image"])
                except Exception as e:
                    logger.debug(f"Image copy failed: {e}")
        except:
            pass
    
    def _copy_drawings(self, source_page, output_page):
        """Copy drawings from source page to output page."""
        try:
            for path in source_page.get_drawings():
                items = path.get("items", [])
                rect = path.get("rect")
                color = path.get("color", (0, 0, 0))
                fill = path.get("fill")
                width = path.get("width", 0.5)
                
                for item in items:
                    if item[0] == "l":  # Line
                        output_page.draw_line(item[1], item[2], color=color, width=width)
                    elif item[0] == "re":  # Rectangle
                        output_page.draw_rect(fitz.Rect(item[1]), color=color, fill=fill, width=width)
                    elif item[0] == "c":  # Curve
                        try:
                            output_page.draw_bezier(item[1], item[2], item[3], item[4], color=color, width=width)
                        except:
                            pass
        except:
            pass
    
    def render_text(self, document: Document, output_path: str):
        """Render as plain text."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in document.segments:
                for block in segment.blocks:
                    text = block.translated_text or block.source_text
                    f.write(text + "\n\n")
    
    def render_markdown(self, document: Document, output_path: str):
        """Render as Markdown."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in document.segments:
                for block in segment.blocks:
                    text = block.translated_text or block.source_text
                    block_type = block.metadata.get("block_type", "paragraph") if block.metadata else "paragraph"
                    
                    if block_type == "title":
                        f.write(f"# {text}\n\n")
                    elif block_type == "heading":
                        f.write(f"## {text}\n\n")
                    elif block_type in ["math_content", "math"]:
                        f.write(f"$$\n{text}\n$$\n\n")
                    else:
                        f.write(f"{text}\n\n")

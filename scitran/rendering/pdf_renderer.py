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
from .font_resolver import resolve_font_for_language

logger = logging.getLogger(__name__)

# Font mapping from common PDF fonts to PyMuPDF base14 fonts
FONT_MAP = {
    # Times family
    "times": "times-roman",
    "times new roman": "times-roman",
    "timesnewroman": "times-roman",
    "timesnewromanps": "times-roman",
    "timesroman": "times-roman",
    "georgia": "times-roman",
    "garamond": "times-roman",
    # Helvetica/Arial family  
    "arial": "helv",
    "helvetica": "helv",
    "arialmt": "helv",
    "helveticaneue": "helv",
    "verdana": "helv",
    "tahoma": "helv",
    "calibri": "helv",
    "segoe": "helv",
    "roboto": "helv",
    # Courier family
    "courier": "cour",
    "courier new": "cour",
    "couriernew": "cour",
    "consolas": "cour",
    "menlo": "cour",
    "inconsolata": "cour",
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
    
    def __init__(
        self,
        strict_mode: bool = False,
        embed_fonts: bool = True,
        font_dir: Optional[str] = None,
        font_files: Optional[List[str]] = None,
        font_priority: Optional[List[str]] = None,
        # PHASE 3.2: Overflow handling
        overflow_strategy: str = "smart",  # "smart", "shrink", "expand", "append_pages", "marker+append_pages"
        min_font_size: float = 8.0,  # Minimum readable font size (never shrink below this)
        min_lineheight: float = 0.95,  # Minimum line height multiplier
        # STEP 7: Font resolution for non-Latin scripts
        target_lang: Optional[str] = None,  # Target language for font resolution
        download_fonts: bool = True,  # Download fonts if missing
    ):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        self.default_font = "helv"
        self.default_fontsize = 11
        self.temp_files = []  # Track temp files for cleanup
        self.strict_mode = strict_mode
        self.embed_fonts = embed_fonts
        self.font_dir = font_dir
        self.font_files = font_files or []
        self.font_priority = [f.lower().replace(" ", "") for f in font_priority] if font_priority else []
        
        # PHASE 3.2: Overflow configuration
        self.overflow_strategy = overflow_strategy
        self.min_font_size = min_font_size
        self.min_lineheight = min_lineheight
        self.overflow_report = []  # Track overflow events
        
        # STEP 7: Font resolution for non-Latin scripts
        self.target_lang = target_lang
        self.download_fonts = download_fonts
        self._resolved_font: Optional[str] = None
        if target_lang and download_fonts:
            try:
                font_path = resolve_font_for_language(target_lang, download_enabled=True)
                if font_path and font_path.exists():
                    self._resolved_font = str(font_path)
                    logger.info(f"Resolved font for {target_lang}: {font_path.name}")
            except Exception as e:
                logger.warning(f"Font resolution failed for {target_lang}: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        self.temp_files = []
    
    def save_overflow_report(self, output_path: str):
        """Save overflow report to JSON file (PHASE 3.2)."""
        if not self.overflow_report:
            return
        
        import json
        report_path = output_path.replace('.pdf', '_overflow_report.json')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_overflows": len(self.overflow_report),
                    "strategy": self.overflow_strategy,
                    "events": self.overflow_report
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Overflow report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save overflow report: {e}")
    
    def render_pdf(self, document: Document, output_path: str, source_pdf: str = None):
        """Main entry point for PDF rendering."""
        if source_pdf:
            return self.render_with_layout(source_pdf, document, output_path)
        return self.render_simple(document, output_path)
    
    def _register_custom_font(self, family: str, bold: bool, italic: bool) -> Optional[str]:
        """Attempt to register a custom TTF/OTF font from explicit files first, then font_dir."""
        import glob

        def candidate_paths():
            # explicit files take precedence
            for f in self.font_files:
                yield f
            if self.font_dir:
                for f in glob.glob(f"{self.font_dir}/**/*.ttf", recursive=True):
                    yield f
                for f in glob.glob(f"{self.font_dir}/**/*.otf", recursive=True):
                    yield f

        style_tokens = []
        if bold:
            style_tokens.append("bold")
        if italic:
            style_tokens.append("italic")
        family_lower = family.lower().replace(" ", "")

        for path in candidate_paths():
            try:
                name = Path(path).stem.lower().replace(" ", "")
                if self.font_priority:
                    # Enforce priority ordering if provided
                    if not any(pref in name for pref in self.font_priority):
                        continue
                if family_lower and family_lower not in name:
                    # allow loose match if priority is set
                    if not self.font_priority:
                        continue
                if not all(tok in name for tok in style_tokens):
                    continue
                fontname = fitz.Font(fontfile=path).name
                return fontname
            except Exception:
                continue
        return None

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
        
        # Try custom font registration if a font_dir is provided
        custom_font = self._register_custom_font(base_font, is_bold, is_italic) if self.font_dir else None
        if custom_font:
            return custom_font, is_bold, is_italic

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
        elif base_font == "symb":
            return "symb", is_bold, is_italic
        elif base_font == "zadb":
            return "zadb", is_bold, is_italic
        # Fallback to a readable sans font but keep style flags
        if is_bold and is_italic:
            return "hebo", is_bold, is_italic
        if is_bold:
            return "hebo", is_bold, is_italic
        if is_italic:
            return "heob", is_bold, is_italic
        
        # For unknown fonts, return with style flags
        return base_font, is_bold, is_italic
    
    def _get_unicode_font(self, base_fontname: str, is_bold: bool, is_italic: bool) -> Optional[str]:
        """
        Get a Unicode-capable font for special characters (apostrophes, accents, etc.).
        Base14 fonts don't support Unicode properly, so we need to use system fonts.
        
        Returns:
            Font name string that PyMuPDF can use, or None if not available
        """
        import platform
        
        # Try to register a Unicode-capable font from system
        # Common Unicode fonts: Times New Roman, Arial, Helvetica (system versions)
        unicode_fonts = []
        
        if platform.system() == "Darwin":  # macOS
            # macOS has good Unicode support in system fonts
            if "times" in base_fontname.lower() or "tiro" in base_fontname.lower():
                unicode_fonts = ["Times-Roman", "TimesNewRoman", "Times"]
            else:
                unicode_fonts = ["Helvetica", "Arial", "HelveticaNeue"]
        elif platform.system() == "Windows":
            if "times" in base_fontname.lower() or "tiro" in base_fontname.lower():
                unicode_fonts = ["Times New Roman", "TimesNewRoman"]
            else:
                unicode_fonts = ["Arial", "Helvetica"]
        else:  # Linux
            if "times" in base_fontname.lower() or "tiro" in base_fontname.lower():
                unicode_fonts = ["Liberation Serif", "Times New Roman", "DejaVu Serif"]
            else:
                unicode_fonts = ["Liberation Sans", "Arial", "DejaVu Sans"]
        
        # Try to register one of these fonts
        for font_name in unicode_fonts:
            try:
                # Try to register the font if we have font_dir
                registered = self._register_custom_font(font_name, is_bold, is_italic)
                if registered:
                    return registered
            except:
                pass
        
        # Fallback: try to use a font file if available
        if self.font_dir or self.font_files:
            try:
                registered = self._register_custom_font(base_fontname, is_bold, is_italic)
                if registered:
                    return registered
            except:
                pass
        
        # If no Unicode font available, return None (will use base14 font)
        # PyMuPDF will handle Unicode characters by falling back to glyph substitution
        return None
    
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
                # Use left align for simple rendering (no alignment info available)
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
        
        # Extract text blocks with detailed info (preserve ligatures/spaces)
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
        
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
        """
        Remove source text from specific regions using non-destructive redaction.
        
        CRITICAL: Uses fill=None to remove text without painting over graphics.
        This ensures figures, tables, diagrams, and images remain intact.
        
        Combined with apply_redactions(images=0, graphics=0), this guarantees
        that only text is removed while all vector graphics and images are preserved.
        """
        redact_count = 0
        
        for block in blocks:
            # Skip protected blocks (equation only - tables/figures are now translatable)
            # STEP 2: TABLE and FIGURE are no longer automatically protected
            if block.block_type == BlockType.EQUATION:
                continue
            # Check if block is protected (handle both dict and TranslationMetadata)
            if block.metadata:
                if isinstance(block.metadata, dict):
                    if block.metadata.get("protected_reason"):
                        continue
                elif hasattr(block.metadata, "protected_reason") and block.metadata.protected_reason:
                    continue
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
                
            # Add redaction annotation with NO fill (non-destructive)
            # CRITICAL: fill=None means remove text only, don't paint over graphics
            page.add_redact_annot(
                rect, 
                fill=None,  # No fill - preserves underlying graphics/images
                text="",  # No replacement text
            )
            redact_count += 1
        
        # Apply all redactions - CRITICAL: Must not remove vector graphics or images
        # Parameters: images, graphics, text (0=keep, 1=remove, 2=remove+mark)
        if redact_count > 0:
            page.apply_redactions(
                images=0,    # 0 = Keep images (don't remove)
                graphics=0   # 0 = Keep vector graphics (DON'T REMOVE)
            )
            logger.info(f"Applied {redact_count} redactions on page (text only, preserving graphics)")
    
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
                
                # Collect preservable blocks (equations only - tables/figures now translatable)
                # STEP 2: Only preserve EQUATION blocks (math formulas)
                # TABLE and FIGURE text will be translated, so don't stamp them
                if block.block_type == BlockType.EQUATION:
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
        
        # #region agent log
        try:
            import json
            from datetime import datetime
            # Use configurable debug log path or default to .cache
            debug_log_dir = Path.home() / ".scitrans" / "logs"
            debug_log_dir.mkdir(parents=True, exist_ok=True)
            DEBUG_LOG_PATH = debug_log_dir / "debug.log"
            log_entry = {
                "sessionId": "translation_debug",
                "runId": "render_pages",
                "hypothesisId": "H2",
                "location": "pdf_renderer.py:render_with_layout",
                "message": "Page processing start",
                "data": {
                    "total_pages": total_pages_in_doc,
                    "pages_with_blocks": sorted(blocks_by_page.keys()),
                    "translated_count": translated_count,
                    "skipped_no_translation": skipped_no_translation
                },
                "timestamp": datetime.now().isoformat()
            }
            with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Process each page
        pages_processed = 0
        for page_num in range(total_pages_in_doc):
            page = doc[page_num]
            
            # CRITICAL: Always stamp preserved blocks FIRST, before any redaction
            # This ensures images/tables/formulas are preserved even if text overlaps
            if page_num in preserve_by_page:
                logger.info(f"Page {page_num + 1}/{total_pages_in_doc}: preserving {len(preserve_by_page[page_num])} non-translatable blocks (images/tables/formulas)")
                self._stamp_preserved_blocks(page, source_doc[page_num], preserve_by_page[page_num])
            
            if page_num in blocks_by_page:
                page_blocks = blocks_by_page[page_num]
                logger.info(f"Page {page_num + 1}/{total_pages_in_doc}: processing {len(page_blocks)} translated blocks")
                
                # Step 1: Redact (remove) source text in translation regions
                # Note: This happens AFTER stamping, so preserved blocks are safe
                self._redact_text_from_page(page, page_blocks)
                
                # Step 2: Insert translated text
                for block in page_blocks:
                    self._insert_text_block(page, block)
                pages_processed += 1
            else:
                logger.info(f"Page {page_num + 1}/{total_pages_in_doc}: no translated blocks (preserved blocks already stamped)")
        
        logger.info(f"Processed {pages_processed} pages with translations out of {total_pages_in_doc} total pages")
        
        # STEP 5: Create overflow pages if strategy is append_pages or smart (fallback)
        if self.overflow_strategy in ["append_pages", "marker+append_pages", "smart"]:
            overflow_blocks = [e for e in self.overflow_report if "full_text" in e]
            if overflow_blocks:
                logger.info(f"Creating overflow pages for {len(overflow_blocks)} blocks")
                self._create_overflow_pages(doc, overflow_blocks)
        
        # Save output (embed fonts if requested)
        save_opts = {"garbage": 4, "deflate": True}
        if self.embed_fonts:
            # Subset fonts to preserve styling as much as possible
            doc.subset_fonts()
        doc.save(output_path, **save_opts)
        doc.close()
        source_doc.close()
        
        # PHASE 3.2: Save overflow report if any overflows occurred
        if self.overflow_report:
            self.save_overflow_report(output_path)
            logger.warning(f"Rendering completed with {len(self.overflow_report)} overflow events")
        
        # Cleanup temp
        self.cleanup()
        
        logger.info(f"PDF rendered to {output_path}")
    
    def _insert_text_block(self, page, block: Block):
        """Insert a translated text block with proper overflow handling (PHASE 3.2)."""
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
        
        # Get font info with enhanced styling
        fontname, is_bold, is_italic = self._get_font_name(block.font)
        
        # Use enhanced font properties if available
        line_height_multiplier = 1.2  # Default
        if block.font and block.font.line_height:
            line_height_multiplier = block.font.line_height
        
        # Get text FIRST before any Unicode checks
        text = block.translated_text
        if not text or not text.strip():
            return
        
        # Ensure text is properly encoded as UTF-8 string (Python 3 default)
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        elif not isinstance(text, str):
            text = str(text)
        
        # CRITICAL FIX: Use Unicode-capable font for special characters (apostrophes, accents)
        # Check if text contains Unicode characters that need special font support
        # Also check for common apostrophe characters that might be replaced
        apostrophe_chars = ["'", "'", "'", "`", "`"]
        has_unicode = any(ord(c) > 127 for c in text) or any(c in text for c in apostrophe_chars)
        if has_unicode and self.download_fonts:
            unicode_font = self._get_unicode_font(fontname, is_bold, is_italic)
            if unicode_font:
                fontname = unicode_font
                logger.debug(f"Using Unicode font {fontname} for block {block.block_id} (Unicode chars detected)")
        
        # Normalize apostrophes - ensure they're proper Unicode apostrophes
        # Replace common problematic apostrophes with proper ones
        text = text.replace("'", "'")  # Replace straight apostrophe with curly
        text = text.replace("'", "'")  # Replace right single quote with apostrophe
        
        # Detect alignment
        alignment = 0  # Default: left align (0=left, 1=center, 2=right, 3=justify)
        if block.font and hasattr(block.font, 'alignment') and block.font.alignment:
            align_str = block.font.alignment.lower()
            if align_str == "center":
                alignment = 1
            elif align_str == "right":
                alignment = 2
            elif align_str == "justify":
                alignment = 3
        else:
            # Infer alignment from block position
            page_width = page.rect.width
            block_center = (rect.x0 + rect.x1) / 2
            page_center = page_width / 2
            if abs(block_center - page_center) < page_width * 0.1:
                alignment = 1  # Center
            elif rect.x0 > page_width * 0.7:
                alignment = 2  # Right
        
        # Get font size - preserve original size for consistency
        if block.font and block.font.size and block.font.size > 0:
            fontsize = block.font.size
            # Don't scale down too aggressively - preserve original size when possible
        else:
            bbox_height = rect.height
            num_lines = max(1, block.translated_text.count('\n') + 1)
            estimated_size = bbox_height / num_lines / 1.2
            # Use a reasonable default that matches typical academic text
            fontsize = max(10, min(12, estimated_size))  # Default to 10-12pt for readability
        
        # Get color
        color = (0, 0, 0)
        if block.font and block.font.color:
            try:
                hex_color = block.font.color.lstrip('#')
                if len(hex_color) == 6:
                    color = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
            except:
                pass
        
        # Text already assigned earlier in the function, no need to reassign
        
        # #region agent log
        try:
            import json
            from datetime import datetime
            # Use configurable debug log path or default to .cache
            debug_log_dir = Path.home() / ".scitrans" / "logs"
            debug_log_dir.mkdir(parents=True, exist_ok=True)
            DEBUG_LOG_PATH = debug_log_dir / "debug.log"
            log_entry = {
                "sessionId": "translation_debug",
                "runId": "render_text",
                "hypothesisId": "H1",
                "location": "pdf_renderer.py:_insert_text_block",
                "message": "Before text normalization",
                "data": {
                    "block_id": block.block_id,
                    "page": block.bbox.page if block.bbox else None,
                    "text_length": len(text),
                    "has_apostrophe": "'" in text or "'" in text or "'" in text,
                    "has_question_mark": "?" in text,
                    "line_breaks_count": text.count('\n'),
                    "text_preview": text[:100] if len(text) > 100 else text,
                    "text_type": type(text).__name__
                },
                "timestamp": datetime.now().isoformat()
            }
            with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Normalize line breaks: replace multiple newlines with single space within paragraphs
        # But preserve intentional paragraph breaks (double newlines)
        import re
        # Split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        normalized_paragraphs = []
        for para in paragraphs:
            # Replace single newlines within paragraph with space
            normalized_para = re.sub(r'\n+', ' ', para.strip())
            normalized_paragraphs.append(normalized_para)
        text = '\n\n'.join(normalized_paragraphs)
        
        # #region agent log
        try:
            log_entry = {
                "sessionId": "translation_debug",
                "runId": "render_text",
                "hypothesisId": "H1",
                "location": "pdf_renderer.py:_insert_text_block",
                "message": "After text normalization",
                "data": {
                    "block_id": block.block_id,
                    "text_length": len(text),
                    "line_breaks_count": text.count('\n'),
                    "text_preview": text[:100] if len(text) > 100 else text
                },
                "timestamp": datetime.now().isoformat()
            }
            with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Try inserting with progressively smaller sizes
        # Use tighter line height for better text fitting
        line_height = 1.15 if fontsize > 12 else 1.1
        # More gradual size reduction to preserve readability
        sizes_to_try = [
            fontsize,  # Original size
            fontsize * 0.95,  # 5% smaller
            fontsize * 0.9,   # 10% smaller
            fontsize * 0.85,  # 15% smaller
            fontsize * 0.8,   # 20% smaller
            max(self.min_font_size, fontsize * 0.75)  # 25% smaller or min
        ]
        
        for size in sizes_to_try:
            try:
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontsize=size,
                    fontname=fontname,
                    color=color,
                    align=alignment,
                    lineheight=line_height * self.min_lineheight,
                    render_mode=0,
                )
                
                if rc >= 0:
                    # Text fit successfully
                    logger.debug(f"✓ Inserted block {block.block_id}: size={size:.1f}")
                    return
            except Exception as e:
                logger.debug(f"Insert attempt failed at size {size}: {e}")
                continue
        
        # PHASE 3.2: All size attempts failed - handle overflow
        logger.warning(f"Overflow detected for block {block.block_id}, strategy={self.overflow_strategy}")
        
        overflow_event = {
            "block_id": block.block_id,
            "page": block.bbox.page if block.bbox else None,
            "text_length": len(text),
            "strategy": self.overflow_strategy,
        }
        self.overflow_report.append(overflow_event)
        
        if self.overflow_strategy == "smart":
            # Smart strategy: Try multiple approaches in order of preference
            success = self._smart_overflow_handling(page, rect, text, fontname, fontsize, color, alignment, line_height, block)
            if success:
                return  # Successfully handled overflow
        
        if self.overflow_strategy == "shrink":
            # Force insert at minimum size (may truncate)
            try:
                page.insert_textbox(
                    rect, text, fontsize=self.min_font_size,
                    fontname=fontname, color=color, align=alignment,
                    lineheight=self.min_lineheight
                )
                logger.warning(f"Forced insert at min size for {block.block_id}")
            except Exception as e:
                logger.error(f"Failed to insert even at min size: {e}")
                # Last resort: try with default Base14 font (always available)
                try:
                    # Use Base14 font that's guaranteed to exist
                    fallback_font = "helv"  # Helvetica is always available in PyMuPDF
                    page.insert_textbox(
                        rect, text, fontsize=self.min_font_size,
                        fontname=fallback_font, color=color, align=alignment,
                        lineheight=self.min_lineheight
                    )
                    logger.warning(f"Inserted block {block.block_id} with fallback font '{fallback_font}'")
                except Exception as e2:
                    logger.error(f"Failed even with fallback font for {block.block_id}: {e2}")
                    # Mark block as failed but don't crash
                    overflow_event["error"] = str(e2)
                    overflow_event["status"] = "failed"
        
        elif self.overflow_strategy == "expand":
            # Try expanding rect downward
            expanded_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, min(rect.y1 + 100, page.rect.height))
            try:
                rc = page.insert_textbox(
                    expanded_rect, text, fontsize=fontsize * 0.8,
                    fontname=fontname, color=color, align=alignment,
                    lineheight=line_height
                )
                if rc >= 0:
                    logger.info(f"Overflow resolved by expanding rect for {block.block_id}")
                else:
                    logger.warning(f"Expand strategy still overflowed for {block.block_id}")
            except Exception as e:
                logger.error(f"Expand strategy failed: {e}")
        
        elif self.overflow_strategy in ["append_pages", "marker+append_pages"]:
            # Insert marker in original location
            if self.overflow_strategy == "marker+append_pages":
                marker = f"(continued on overflow page)"
                try:
                    page.insert_textbox(
                        rect, marker, fontsize=fontsize * 0.8,
                        fontname=fontname, color=(0.5, 0, 0), align=alignment
                    )
                except:
                    pass
            
            # Store full text for overflow page (will be added in render_with_layout)
            overflow_event["full_text"] = text
            overflow_event["fontsize"] = fontsize
            overflow_event["fontname"] = fontname
            overflow_event["color"] = color
            logger.info(f"Marked block {block.block_id} for overflow page")
        
        else:
            logger.error(f"Unknown overflow strategy: {self.overflow_strategy}")
    
    def _smart_overflow_handling(
        self, page, rect, text: str, fontname: str, fontsize: float,
        color: Tuple[float, float, float], alignment: int, line_height: float, block: Block
    ) -> bool:
        """
        Smart overflow handling: Try multiple strategies to prevent overflow.
        
        Returns:
            True if overflow was successfully handled, False otherwise
        """
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Strategy 1: Try expanding downward (most common case)
        space_below = page_height - rect.y1
        if space_below > 20:  # At least 20pt of space below
            # Calculate how much space we need
            estimated_lines = len(text.split('\n')) or max(1, len(text) // 50)
            estimated_height = estimated_lines * fontsize * line_height * 1.2
            needed_height = estimated_height - rect.height
            
            if needed_height > 0 and space_below >= needed_height * 0.8:  # 80% of needed space available
                expanded_rect = fitz.Rect(
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    min(rect.y1 + needed_height * 1.2, page_height - 10)  # Add 20% buffer, leave 10pt margin
                )
                try:
                    rc = page.insert_textbox(
                        expanded_rect, text, fontsize=fontsize,
                        fontname=fontname, color=color, align=alignment,
                        lineheight=line_height
                    )
                    if rc >= 0:
                        logger.info(f"✓ Smart overflow: Expanded downward for {block.block_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Downward expansion failed: {e}")
        
        # Strategy 2: Try expanding upward (if space available above)
        space_above = rect.y0
        if space_above > 20:  # At least 20pt of space above
            estimated_lines = len(text.split('\n')) or max(1, len(text) // 50)
            estimated_height = estimated_lines * fontsize * line_height * 1.2
            needed_height = estimated_height - rect.height
            
            if needed_height > 0 and space_above >= needed_height * 0.8:
                expanded_rect = fitz.Rect(
                    rect.x0,
                    max(10, rect.y0 - needed_height * 1.2),  # Expand upward, leave 10pt margin
                    rect.x1,
                    rect.y1
                )
                try:
                    rc = page.insert_textbox(
                        expanded_rect, text, fontsize=fontsize,
                        fontname=fontname, color=color, align=alignment,
                        lineheight=line_height
                    )
                    if rc >= 0:
                        logger.info(f"✓ Smart overflow: Expanded upward for {block.block_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Upward expansion failed: {e}")
        
        # Strategy 3: Try expanding both directions (if space available)
        total_available = space_above + space_below
        if total_available > 40:  # At least 40pt total space
            estimated_lines = len(text.split('\n')) or max(1, len(text) // 50)
            estimated_height = estimated_lines * fontsize * line_height * 1.2
            needed_height = estimated_height - rect.height
            
            if needed_height > 0 and total_available >= needed_height * 0.8:
                # Distribute expansion proportionally
                expand_up = min(space_above * 0.8, needed_height * 0.5)
                expand_down = min(space_below * 0.8, needed_height * 0.5)
                
                expanded_rect = fitz.Rect(
                    rect.x0,
                    max(10, rect.y0 - expand_up),
                    rect.x1,
                    min(page_height - 10, rect.y1 + expand_down)
                )
                try:
                    rc = page.insert_textbox(
                        expanded_rect, text, fontsize=fontsize,
                        fontname=fontname, color=color, align=alignment,
                        lineheight=line_height
                    )
                    if rc >= 0:
                        logger.info(f"✓ Smart overflow: Expanded both directions for {block.block_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Bidirectional expansion failed: {e}")
        
        # Strategy 4: Try slightly reducing font size (but not below readable threshold)
        readable_size = max(8.0, fontsize * 0.85)  # Never go below 8pt
        if readable_size >= 8.0:
            for reduced_size in [fontsize * 0.9, fontsize * 0.85, readable_size]:
                try:
                    rc = page.insert_textbox(
                        rect, text, fontsize=reduced_size,
                        fontname=fontname, color=color, align=alignment,
                        lineheight=line_height * 0.95  # Tighter line height
                    )
                    if rc >= 0:
                        logger.info(f"✓ Smart overflow: Reduced font to {reduced_size:.1f}pt for {block.block_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Font reduction failed at {reduced_size}: {e}")
                    continue
        
        # Strategy 5: Fall back to append_pages (preserve full text, never truncate)
        logger.info(f"Smart overflow: Using append_pages fallback for {block.block_id}")
        overflow_event = {
            "block_id": block.block_id,
            "page": block.bbox.page if block.bbox else None,
            "text_length": len(text),
            "strategy": "append_pages",
            "full_text": text,
            "fontsize": fontsize,
            "fontname": fontname,
            "color": color,
        }
        # Update the overflow event in the report
        for i, event in enumerate(self.overflow_report):
            if event.get("block_id") == block.block_id:
                self.overflow_report[i] = overflow_event
                break
        
        # Insert marker in original location
        marker = f"(→ voir page suivante)"
        try:
            page.insert_textbox(
                rect, marker, fontsize=fontsize * 0.7,
                fontname=fontname, color=(0.5, 0, 0), align=alignment
            )
        except:
            pass
        
        return True  # Handled via append_pages
    
    def _create_overflow_pages(self, doc, overflow_blocks: List[Dict[str, Any]]):
        """
        Create additional pages for overflow text (STEP 5).
        
        Args:
            doc: PyMuPDF document
            overflow_blocks: List of overflow event dicts with full_text
        """
        if not overflow_blocks:
            return
        
        # Group by page for better organization
        by_page = {}
        for event in overflow_blocks:
            page_num = event.get("page", 0)
            if page_num not in by_page:
                by_page[page_num] = []
            by_page[page_num].append(event)
        
        # Create overflow pages
        for page_num, events in sorted(by_page.items()):
            # Create new page
            overflow_page = doc.new_page(
                width=doc[0].rect.width if len(doc) > 0 else 595,
                height=doc[0].rect.height if len(doc) > 0 else 842
            )
            
            # Add header
            header_rect = fitz.Rect(50, 50, overflow_page.rect.width - 50, 80)
            overflow_page.insert_textbox(
                header_rect,
                f"Suite du texte de la page {page_num + 1}",
                fontsize=14,
                fontname="helv",  # Fixed typo: was "hebo", should be "helv"
                color=(0.5, 0, 0),
                align=1  # Center
            )
            
            # Place overflow texts
            y_position = 100
            margin = 50
            
            for event in events:
                text = event.get("full_text", "")
                if not text:
                    continue
                
                fontsize = event.get("fontsize", 11)
                fontname = event.get("fontname", "helv")
                color = event.get("color", (0, 0, 0))
                
                # Calculate available space
                available_height = overflow_page.rect.height - y_position - margin
                
                if available_height < 50:
                    # Need another page
                    overflow_page = doc.new_page(
                        width=doc[0].rect.width,
                        height=doc[0].rect.height
                    )
                    y_position = 50
                    available_height = overflow_page.rect.height - y_position - margin
                
                # Insert text
                rect = fitz.Rect(
                    margin,
                    y_position,
                    overflow_page.rect.width - margin,
                    y_position + available_height
                )
                
                try:
                    rc = overflow_page.insert_textbox(
                        rect,
                        text,
                        fontsize=fontsize * 0.9,  # Slightly smaller
                        fontname=fontname,
                        color=color,
                        align=0,  # Left align
                        lineheight=1.2
                    )
                    
                    # Estimate space used (rough)
                    lines_used = text.count('\n') + 1
                    y_position += lines_used * fontsize * 1.2 + 20
                    
                except Exception as e:
                    logger.error(f"Failed to insert overflow text: {e}")
                    y_position += 50
        
        logger.info(f"Created overflow pages, total pages now: {len(doc)}")
    
    def _stamp_preserved_blocks(self, target_page, source_page, blocks: List[Block]):
        """Stamp source regions for tables/equations/figures - VECTOR ONLY (no rasterization).
        
        NOTE: With proper redaction (graphics=0), this should rarely be needed.
        This is a safety fallback only.
        """
        for block in blocks:
            if not block.bbox:
                continue
            rect = fitz.Rect(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
            if rect.is_empty or rect.is_infinite:
                continue
            try:
                # VECTOR stamping using show_pdf_page (NO RASTERIZATION)
                # This preserves vector graphics perfectly
                target_page.show_pdf_page(
                    rect,  # Target position
                    source_page.parent,  # Source PDF
                    source_page.number,  # Source page number
                    clip=rect,  # Clip to this region
                    keep_proportion=False,
                    overlay=True  # Overlay on existing content
                )
                logger.debug(f"Vector-stamped preserved region for {block.block_id} ({block.block_type})")
            except Exception as e:
                logger.warning(f"Failed to vector-stamp block {block.block_id}: {e}")
    
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
                    # Get block_type from metadata (handle both dict and TranslationMetadata)
                    if block.metadata:
                        if isinstance(block.metadata, dict):
                            block_type = block.metadata.get("block_type", "paragraph")
                        elif hasattr(block.metadata, "block_type"):
                            block_type = block.metadata.block_type or "paragraph"
                        else:
                            block_type = "paragraph"
                    else:
                        block_type = "paragraph"
                    
                    if block_type == "title":
                        f.write(f"# {text}\n\n")
                    elif block_type == "heading":
                        f.write(f"## {text}\n\n")
                    elif block_type in ["math_content", "math"]:
                        f.write(f"$$\n{text}\n$$\n\n")
                    else:
                        f.write(f"{text}\n\n")

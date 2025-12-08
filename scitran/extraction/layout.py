"""Layout detection and analysis."""

from typing import List, Dict, Optional
from ..core.models import Block, BoundingBox


class LayoutDetector:
    """Detect and analyze document layout."""
    
    def __init__(self, use_yolo: bool = False):
        """
        Initialize layout detector.
        
        Args:
            use_yolo: Whether to use YOLO-based detection (requires model)
        """
        self.use_yolo = use_yolo
        self.yolo_model = None
        
        if use_yolo:
            try:
                from .yolo import load_yolo_model
                self.yolo_model = load_yolo_model()
            except ImportError:
                print("Warning: YOLO not available, using heuristic detection")
                self.use_yolo = False
    
    def detect_layout(self, blocks: List[Block]) -> List[Block]:
        """
        Detect layout elements in blocks.
        
        Args:
            blocks: List of document blocks
            
        Returns:
            Blocks with enhanced layout information
        """
        if self.use_yolo and self.yolo_model:
            return self._detect_with_yolo(blocks)
        else:
            return self._detect_heuristic(blocks)
    
    def _detect_heuristic(self, blocks: List[Block]) -> List[Block]:
        """Heuristic-based layout detection."""
        for block in blocks:
            block_type = self._classify_block_heuristic(block)
            if block.metadata is None:
                block.metadata = {}
            block.metadata["layout_type"] = block_type
            block.metadata["detection_method"] = "heuristic"
        
        return blocks
    
    def _classify_block_heuristic(self, block: Block) -> str:
        """Classify block using heuristics."""
        text = block.source_text.strip()
        
        # Title detection
        if len(text) < 100 and text.isupper():
            return "title"
        
        # Abstract detection
        if text.lower().startswith("abstract"):
            return "abstract"
        
        # Section heading
        if len(text.split()) < 10 and not text.endswith("."):
            return "heading"
        
        # Math/formula detection
        if "$" in text or "\\" in text:
            return "math"
        
        # List detection
        if text.startswith(("•", "-", "*", "–")) or text[0:3].match(r"\d+\."):
            return "list"
        
        # Default to paragraph
        return "paragraph"
    
    def _detect_with_yolo(self, blocks: List[Block]) -> List[Block]:
        """YOLO-based layout detection (placeholder for future implementation)."""
        # This would use a trained YOLO model to detect:
        # - Titles, headings, paragraphs
        # - Figures and captions
        # - Tables and equations
        # - Headers and footers
        
        print("Warning: YOLO detection not fully implemented yet")
        return self._detect_heuristic(blocks)
    
    def analyze_reading_order(self, blocks: List[Block]) -> List[Block]:
        """Determine reading order of blocks."""
        # Sort by page, then by vertical position, then horizontal
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (
                b.bbox.page if b.bbox else 0,
                b.bbox.y if b.bbox else 0,
                b.bbox.x if b.bbox else 0
            )
        )
        
        return sorted_blocks
    
    def detect_columns(self, blocks: List[Block]) -> Dict[int, int]:
        """Detect number of columns per page."""
        page_columns = {}
        
        blocks_by_page = {}
        for block in blocks:
            page = block.bbox.page if block.bbox else 0
            if page not in blocks_by_page:
                blocks_by_page[page] = []
            blocks_by_page[page].append(block)
        
        for page, page_blocks in blocks_by_page.items():
            if not page_blocks:
                page_columns[page] = 1
                continue
            
            # Get x positions
            x_positions = [b.bbox.x for b in page_blocks if b.bbox]
            if not x_positions:
                page_columns[page] = 1
                continue
            
            # Simple clustering: if blocks have very different x positions, likely multiple columns
            unique_x = sorted(set(x_positions))
            
            # Group x positions that are close together (within 50 units)
            groups = []
            current_group = [unique_x[0]]
            
            for x in unique_x[1:]:
                if x - current_group[-1] < 50:
                    current_group.append(x)
                else:
                    groups.append(current_group)
                    current_group = [x]
            
            if current_group:
                groups.append(current_group)
            
            page_columns[page] = len(groups)
        
        return page_columns

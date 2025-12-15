"""Layout detection and analysis."""

from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from ..core.models import Block, BoundingBox

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Detect and analyze document layout."""
    
    def __init__(
        self, 
        use_yolo: bool = False,
        yolo_model_path: Optional[str] = None,
        yolo_device: Optional[str] = None,
        yolo_conf_threshold: float = 0.25
    ):
        """
        Initialize layout detector.
        
        Args:
            use_yolo: Whether to use YOLO-based detection (requires model)
            yolo_model_path: Path to YOLO model file (.pt). If None, uses default.
            yolo_device: Device to use ('cuda', 'cpu', 'mps'). Auto-detects if None.
            yolo_conf_threshold: Confidence threshold for YOLO detections
        """
        self.use_yolo = use_yolo
        self.yolo_model = None
        self.yolo_device = yolo_device
        self.yolo_conf_threshold = yolo_conf_threshold
        
        if use_yolo:
            try:
                from .yolo import load_yolo_model
                self.yolo_model = load_yolo_model(
                    model_path=yolo_model_path,
                    device=yolo_device
                )
                if self.yolo_model is None:
                    logger.warning("YOLO model not available, falling back to heuristic detection")
                    self.use_yolo = False
                else:
                    logger.info("YOLO model loaded successfully")
            except ImportError as e:
                logger.warning(f"YOLO not available: {e}. Using heuristic detection")
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
    
    def _detect_with_yolo(self, blocks: List[Block], page_images: Optional[Dict[int, Any]] = None) -> List[Block]:
        """
        YOLO-based layout detection.
        
        Args:
            blocks: List of document blocks
            page_images: Optional dictionary mapping page numbers to images (numpy arrays or PIL Images)
        
        Returns:
            Blocks with YOLO-detected layout information
        """
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded, falling back to heuristic")
            return self._detect_heuristic(blocks)
        
        try:
            from .yolo import detect_layout_elements, map_yolo_classes_to_layout_types
            import numpy as np
            
            # Group blocks by page
            blocks_by_page: Dict[int, List[Block]] = {}
            for block in blocks:
                page = block.bbox.page if block.bbox else 0
                if page not in blocks_by_page:
                    blocks_by_page[page] = []
                blocks_by_page[page].append(block)
            
            # Process each page
            for page_num, page_blocks in blocks_by_page.items():
                if not page_blocks:
                    continue
                
                # Get page image if available
                page_image = None
                if page_images and page_num in page_images:
                    page_image = page_images[page_num]
                else:
                    # Try to get image from first block's document context
                    # For now, we'll use heuristic if no image available
                    logger.debug(f"No image available for page {page_num}, using heuristic for that page")
                    for block in page_blocks:
                        if block.metadata is None:
                            block.metadata = {}
                        block.metadata["layout_type"] = self._classify_block_heuristic(block)
                        block.metadata["detection_method"] = "heuristic"
                    continue
                
                # Run YOLO detection on page image
                detections = detect_layout_elements(
                    self.yolo_model,
                    page_image,
                    conf_threshold=self.yolo_conf_threshold
                )
                
                # Map YOLO classes to layout types
                class_mapping = map_yolo_classes_to_layout_types(detections.get('class_names', []))
                
                # Match detections to blocks based on bounding boxes
                detected_boxes = detections.get('boxes', [])
                detected_classes = detections.get('class_names', [])
                detected_scores = detections.get('scores', [])
                
                # For each block, find the best matching YOLO detection
                for block in page_blocks:
                    if block.bbox is None:
                        # No bbox, use heuristic
                        if block.metadata is None:
                            block.metadata = {}
                        block.metadata["layout_type"] = self._classify_block_heuristic(block)
                        block.metadata["detection_method"] = "heuristic"
                        continue
                    
                    # Find best matching detection by IoU
                    best_match_idx = None
                    best_iou = 0.0
                    
                    block_bbox = [
                        block.bbox.x0,
                        block.bbox.y0,
                        block.bbox.x1,
                        block.bbox.y1
                    ]
                    
                    for idx, det_bbox in enumerate(detected_boxes):
                        iou = self._calculate_iou(block_bbox, det_bbox)
                        if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                            best_iou = iou
                            best_match_idx = idx
                    
                    # Update block metadata
                    if block.metadata is None:
                        block.metadata = {}
                    
                    if best_match_idx is not None:
                        # Use YOLO detection
                        detected_class = detected_classes[best_match_idx]
                        layout_type = class_mapping.get(detected_class, 'paragraph')
                        block.metadata["layout_type"] = layout_type
                        block.metadata["detection_method"] = "yolo"
                        block.metadata["yolo_confidence"] = float(detected_scores[best_match_idx])
                        block.metadata["yolo_iou"] = float(best_iou)
                    else:
                        # No good match, use heuristic
                        block.metadata["layout_type"] = self._classify_block_heuristic(block)
                        block.metadata["detection_method"] = "heuristic"
            
            return blocks
        
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}, falling back to heuristic")
            return self._detect_heuristic(blocks)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        try:
            # Box format: [x0, y0, x1, y1]
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calculate union
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            if union_area == 0:
                return 0.0
            
            return inter_area / union_area
        
        except Exception:
            return 0.0
    
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

"""
Core data models for SciTrans-LLMs NEW.

This module defines the fundamental data structures used throughout the system,
implementing a clean, type-safe architecture with validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime
import json
import hashlib


class BlockType(Enum):
    """Types of document blocks with semantic meaning."""
    PARAGRAPH = auto()
    HEADING = auto()
    SUBHEADING = auto()
    EQUATION = auto()
    CODE = auto()
    TABLE = auto()
    FIGURE = auto()
    CAPTION = auto()
    LIST_ITEM = auto()
    REFERENCE = auto()
    FOOTNOTE = auto()
    HEADER = auto()
    FOOTER = auto()
    ABSTRACT = auto()
    TITLE = auto()
    AUTHOR = auto()
    METADATA = auto()


@dataclass
class BoundingBox:
    """Precise coordinate information for layout preservation."""
    x0: float  # Left coordinate
    y0: float  # Top coordinate  
    x1: float  # Right coordinate
    y1: float  # Bottom coordinate
    page: int  # Page number (0-indexed)
    confidence: float = 1.0  # Detection confidence
    
    def area(self) -> float:
        """Calculate area of bounding box."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    def contains(self, other: BoundingBox) -> bool:
        """Check if this box contains another box."""
        return (self.x0 <= other.x0 and self.y0 <= other.y0 and
                self.x1 >= other.x1 and self.y1 >= other.y1 and
                self.page == other.page)
    
    def overlaps(self, other: BoundingBox) -> bool:
        """Check if this box overlaps with another."""
        if self.page != other.page:
            return False
        return not (self.x1 < other.x0 or self.x0 > other.x1 or
                   self.y1 < other.y0 or self.y0 > other.y1)


@dataclass
class FontInfo:
    """Enhanced font information for style preservation."""
    family: str
    size: float
    weight: str = "normal"  # normal, bold, light, medium, semibold, extrabold
    style: str = "normal"   # normal, italic, oblique
    color: str = "#000000"  # RGB hex
    alignment: str = "left"  # left, center, right, justify (inferred from position)
    # Enhanced styling attributes
    line_height: Optional[float] = None  # Line spacing multiplier (e.g., 1.2, 1.5)
    letter_spacing: Optional[float] = None  # Character spacing in points
    word_spacing: Optional[float] = None  # Word spacing in points
    decoration: str = "none"  # none, underline, strikethrough, overline
    # Text formatting
    is_small_caps: bool = False  # Small caps variant
    is_all_caps: bool = False  # All uppercase
    # List formatting (for LIST_ITEM blocks)
    list_style: Optional[str] = None  # bullet, number, letter, roman
    list_indent: Optional[float] = None  # Indentation level
    # Heading hierarchy (for HEADING blocks)
    heading_level: Optional[int] = None  # 1-6 for H1-H6


@dataclass
class MaskInfo:
    """Information about masked content."""
    original: str           # Original content
    placeholder: str        # Placeholder token
    mask_type: str         # Type of mask (formula, code, url, etc.)
    preserve_formatting: bool = False
    
    
@dataclass 
class TranslationMetadata:
    """Metadata for translation quality and traceability."""
    backend: str                          # Translation backend used
    timestamp: datetime                   # When translated
    duration: float                       # Translation time in seconds
    confidence: float = 0.0              # Translation confidence score
    glossary_terms_used: Set[str] = field(default_factory=set)
    context_used: bool = False          # Whether document context was used
    candidates_generated: int = 1        # Number of candidates
    reranking_applied: bool = False     # Whether reranking was applied
    prompt_template: Optional[str] = None  # Prompt template used
    model_name: Optional[str] = None    # Specific model used
    
    # PHASE 2.3: Per-block status tracking
    status: str = "pending"  # pending, translated_ok, translated_fallback, failed, skipped_nontranslatable, render_overflow
    failure_reason: Optional[str] = None  # Reason if failed
    finish_reason: Optional[str] = None  # From backend: "stop", "length", etc.
    was_truncated: bool = False  # True if detected truncation
    retry_count: int = 0  # Number of retries attempted
    
    # Mask restoration tracking (tolerant unmasking)
    restored_masks: int = 0  # Number of masks successfully restored
    missing_placeholders: List[str] = field(default_factory=list)  # Placeholders that couldn't be restored
    

@dataclass
class Block:
    """
    Atomic unit of translation with complete metadata.
    
    This is the fundamental translatable unit in the system,
    containing source text, translations, layout info, and metadata.
    """
    # Core content
    block_id: str                    # Unique identifier
    source_text: str                  # Original text
    translated_text: Optional[str] = None  # Translation
    masked_text: Optional[str] = None      # Text with masks applied
    
    # Classification
    block_type: BlockType = BlockType.PARAGRAPH
    
    # Layout information  
    bbox: Optional[BoundingBox] = None
    font: Optional[FontInfo] = None
    
    # Masking information
    masks: List[MaskInfo] = field(default_factory=list)
    
    # Translation metadata
    metadata: Optional[TranslationMetadata] = None
    
    # Quality scores
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Relationships
    parent_segment_id: Optional[str] = None
    previous_block_id: Optional[str] = None
    next_block_id: Optional[str] = None
    
    @property
    def is_translatable(self) -> bool:
        """
        Check if block should be translated.
        
        NOTE: This is a simplified check. The actual translatability depends on
        PipelineConfig flags (translate_table_text, translate_figure_text).
        The pipeline will check those flags when deciding which blocks to translate.
        
        This property returns True for blocks that CAN be translated (have text content),
        not necessarily SHOULD be translated.
        """
        # Always non-translatable (no text content or structural only)
        always_non_translatable = {
            BlockType.EQUATION,  # LaTeX/math - preserved as-is
            BlockType.CODE,      # Code blocks - preserved as-is
            BlockType.HEADER,    # Page headers - usually not translated
            BlockType.FOOTER,    # Page footers - usually not translated
            BlockType.METADATA,  # Document metadata - structural
        }
        
        # STEP 2: TABLE and FIGURE are now conditionally translatable
        # They are excluded here only if they truly have no translatable text
        # The pipeline will check translate_table_text/translate_figure_text flags
        
        # CAPTION is always translatable (figure/table captions, labels)
        # TITLE, HEADING, SUBHEADING, PARAGRAPH, etc. are translatable
        return self.block_type not in always_non_translatable
    
    @property
    def has_masks(self) -> bool:
        """Check if block has masked content."""
        return len(self.masks) > 0
    
    def get_content_hash(self) -> str:
        """Generate hash of source content for caching."""
        return hashlib.md5(self.source_text.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Block:
        """Create from dictionary."""
        # Make a copy to avoid modifying original
        data = dict(data)
        
        # Handle enum conversion - handle both "PARAGRAPH" and "BlockType.PARAGRAPH" formats
        if 'block_type' in data and isinstance(data['block_type'], str):
            block_type_str = data['block_type']
            # Remove enum class prefix if present
            if '.' in block_type_str:
                block_type_str = block_type_str.split('.')[-1]
            try:
                data['block_type'] = BlockType[block_type_str]
            except KeyError:
                data['block_type'] = BlockType.PARAGRAPH  # Default fallback
        
        # Handle nested objects that might cause issues
        # Remove keys that shouldn't be passed to constructor directly
        keys_to_process = ['bbox', 'font', 'masks', 'metadata']
        for key in keys_to_process:
            if key in data and data[key] is None:
                del data[key]
            elif key == 'bbox' and isinstance(data.get(key), dict):
                data[key] = BoundingBox(**data[key])
            elif key == 'font' and isinstance(data.get(key), dict):
                data[key] = FontInfo(**data[key])
            elif key == 'masks' and isinstance(data.get(key), list):
                data[key] = [MaskInfo(**m) if isinstance(m, dict) else m for m in data[key]]
        
        return cls(**data)


@dataclass
class Segment:
    """
    Logical section of a document containing related blocks.
    
    Segments group related blocks (e.g., a section with its paragraphs)
    to maintain document structure and context.
    """
    segment_id: str
    segment_type: str  # section, chapter, abstract, etc.
    title: Optional[str] = None
    blocks: List[Block] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def translatable_blocks(self) -> List[Block]:
        """Get only translatable blocks."""
        return [b for b in self.blocks if b.is_translatable]
    
    @property
    def total_blocks(self) -> int:
        """Total number of blocks."""
        return len(self.blocks)
    
    @property
    def translated_blocks(self) -> int:
        """Number of translated blocks."""
        return sum(1 for b in self.blocks 
                  if b.translated_text is not None)
    
    def add_block(self, block: Block) -> None:
        """Add a block maintaining relationships."""
        if self.blocks:
            last_block = self.blocks[-1]
            last_block.next_block_id = block.block_id
            block.previous_block_id = last_block.block_id
        block.parent_segment_id = self.segment_id
        self.blocks.append(block)
    
    def get_context_window(self, block_id: str, window_size: int = 3) -> List[Block]:
        """Get surrounding blocks for context."""
        for i, block in enumerate(self.blocks):
            if block.block_id == block_id:
                start = max(0, i - window_size)
                end = min(len(self.blocks), i + window_size + 1)
                return self.blocks[start:end]
        return []


@dataclass
class Document:
    """
    Complete document with all segments, metadata, and glossary.
    
    This is the top-level container for a translation job,
    containing all segments and document-level metadata.
    """
    document_id: str
    source_lang: str = "en"
    target_lang: str = "fr"
    
    # Document structure
    segments: List[Segment] = field(default_factory=list)
    
    # Metadata
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    creation_time: datetime = field(default_factory=datetime.now)
    
    # Document-specific glossary
    glossary_terms: Dict[str, str] = field(default_factory=dict)
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_blocks(self) -> List[Block]:
        """Get all blocks from all segments."""
        blocks = []
        for segment in self.segments:
            blocks.extend(segment.blocks)
        return blocks
    
    @property
    def translatable_blocks(self) -> List[Block]:
        """Get all translatable blocks."""
        return [b for b in self.all_blocks if b.is_translatable]
    
    @property
    def translation_progress(self) -> float:
        """Calculate translation progress percentage."""
        translatable = self.translatable_blocks
        if not translatable:
            return 100.0
        translated = sum(1 for b in translatable if b.translated_text)
        return (translated / len(translatable)) * 100
    
    def add_segment(self, segment: Segment) -> None:
        """Add a segment to the document."""
        self.segments.append(segment)
    
    def get_block_by_id(self, block_id: str) -> Optional[Block]:
        """Find a block by its ID."""
        for block in self.all_blocks:
            if block.block_id == block_id:
                return block
        return None
    
    def get_previous_translations(self, block_id: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Get previous source-translation pairs for context."""
        translations = []
        for block in self.all_blocks:
            if block.block_id == block_id:
                break
            if block.translated_text and block.is_translatable:
                translations.append((block.source_text, block.translated_text))
        return translations[-limit:] if translations else []
    
    def update_stats(self) -> None:
        """Update document statistics."""
        self.stats = {
            'total_segments': len(self.segments),
            'total_blocks': len(self.all_blocks),
            'translatable_blocks': len(self.translatable_blocks),
            'translated_blocks': sum(1 for b in self.translatable_blocks if b.translated_text),
            'translation_progress': self.translation_progress,
            'total_masks': sum(len(b.masks) for b in self.all_blocks),
            'glossary_terms': len(self.glossary_terms),
            'unique_block_types': len(set(b.block_type for b in self.all_blocks))
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        def serialize(obj):
            if isinstance(obj, (datetime, Enum)):
                return str(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj
        
        return json.dumps(asdict(self), default=serialize, indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Reconstruct nested objects
        segments = []
        for seg_data in data.get('segments', []):
            blocks = [Block.from_dict(b) for b in seg_data.get('blocks', [])]
            seg_data['blocks'] = blocks
            segments.append(Segment(**seg_data))
        data['segments'] = segments
        
        # Handle datetime
        if 'creation_time' in data:
            data['creation_time'] = datetime.fromisoformat(data['creation_time'])
            
        return cls(**data)


@dataclass
class TranslationResult:
    """Result of a translation operation with all metadata."""
    document: Document
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0
    backend_used: str = "unknown"
    
    # Quality metrics
    bleu_score: Optional[float] = None
    chrf_score: Optional[float] = None
    glossary_adherence: Optional[float] = None
    layout_preservation: Optional[float] = None
    
    # SPRINT 1: Coverage guarantee metrics
    coverage: float = 1.0  # Ratio of successfully translated blocks (0-1)
    failure_report: Optional[Dict[str, Any]] = None  # Detailed failure info
    validation_result: Optional[Any] = None  # ValidationResult from completeness validator
    
    # Detailed stats
    blocks_translated: int = 0
    blocks_failed: int = 0
    masks_applied: int = 0
    masks_restored: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'success': self.success,
            'duration': self.duration,
            'backend': self.backend_used,
            'progress': self.document.translation_progress,
            'blocks_translated': self.blocks_translated,
            'blocks_failed': self.blocks_failed,
            'quality': {
                'bleu': self.bleu_score,
                'chrf': self.chrf_score,
                'glossary': self.glossary_adherence,
                'layout': self.layout_preservation
            }
        }

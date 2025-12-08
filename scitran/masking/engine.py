"""
Advanced masking engine with validation for SciTrans-LLMs NEW.

This module implements Innovation #1: Terminology-Constrained Translation
through intelligent content masking with post-translation validation.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Pattern
from collections import defaultdict
import logging

from scitran.core.models import Block, Document, MaskInfo

logger = logging.getLogger(__name__)


@dataclass
class MaskPattern:
    """Definition of a masking pattern."""
    name: str
    pattern: Pattern[str]
    priority: int = 0  # Higher priority patterns are applied first
    preserve_formatting: bool = False
    validate_restoration: bool = True
    

@dataclass
class MaskingConfig:
    """Configuration for the masking engine."""
    # Enable/disable specific mask types
    mask_latex: bool = True
    mask_code: bool = True
    mask_urls: bool = True
    mask_emails: bool = True
    mask_numbers: bool = False  # Scientific notation, measurements
    mask_references: bool = True  # Citations like [1], (Smith, 2020)
    mask_acronyms: bool = False
    
    # Validation settings
    strict_validation: bool = True  # Fail if masks can't be restored
    log_validation_errors: bool = True
    
    # Performance settings
    use_cache: bool = True
    parallel_processing: bool = False


class MaskingEngine:
    """
    Advanced masking engine with validation and recovery.
    
    Key improvements over original:
    - Pattern priority system
    - Validation of mask restoration
    - Better LaTeX pattern coverage
    - Nested mask handling
    - Performance optimizations
    """
    
    def __init__(self, config: Optional[MaskingConfig] = None):
        self.config = config or MaskingConfig()
        self.patterns = self._initialize_patterns()
        self.mask_registry: Dict[str, MaskInfo] = {}
        self.mask_counter = defaultdict(int)
        self.validation_errors: List[str] = []
        
    def _initialize_patterns(self) -> List[MaskPattern]:
        """Initialize all masking patterns with priorities."""
        patterns = []
        
        if self.config.mask_latex:
            # LaTeX display environments (highest priority)
            patterns.append(MaskPattern(
                name="latex_environment",
                pattern=re.compile(
                    r'\\begin\{(equation|align|gather|multline|eqnarray|'
                    r'theorem|proof|lemma|proposition|corollary|definition|'
                    r'example|remark|figure|table|algorithm)\*?\}'
                    r'.*?\\end\{\1\*?\}',
                    re.DOTALL
                ),
                priority=100,
                preserve_formatting=True
            ))
            
            # LaTeX display math
            patterns.append(MaskPattern(
                name="latex_display",
                pattern=re.compile(r'\$\$[^$]+\$\$|\\\[[^\]]+\\\]', re.DOTALL),
                priority=95
            ))
            
            # LaTeX inline math
            patterns.append(MaskPattern(
                name="latex_inline",
                pattern=re.compile(r'\$(?!\$)[^$\n]+?\$', re.DOTALL),
                priority=90
            ))
            
            # LaTeX commands (without arguments)
            patterns.append(MaskPattern(
                name="latex_command",
                pattern=re.compile(
                    r'\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|'
                    r'omega|Omega|pi|Pi|sum|int|prod|lim|log|ln|sin|cos|tan|'
                    r'frac|sqrt|partial|nabla|infty|leq|geq|neq|approx|equiv|'
                    r'subset|supset|cap|cup|forall|exists|rightarrow|leftarrow|'
                    r'Rightarrow|Leftarrow|cdot|times|div|pm|mp)\b'
                ),
                priority=85
            ))
            
        if self.config.mask_code:
            # Code blocks with language specification
            patterns.append(MaskPattern(
                name="code_block_lang",
                pattern=re.compile(r'```[\w+#-]*\n.*?\n```', re.DOTALL),
                priority=80,
                preserve_formatting=True
            ))
            
            # Generic code blocks
            patterns.append(MaskPattern(
                name="code_block",
                pattern=re.compile(r'```.*?```', re.DOTALL),
                priority=75,
                preserve_formatting=True
            ))
            
            # Inline code
            patterns.append(MaskPattern(
                name="code_inline",
                pattern=re.compile(r'`[^`\n]+`'),
                priority=70
            ))
            
        if self.config.mask_urls:
            # URLs with protocols
            patterns.append(MaskPattern(
                name="url_full",
                pattern=re.compile(
                    r'(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*'
                    r'[-A-Za-z0-9+&@#/%=~_|]'
                ),
                priority=60
            ))
            
            # DOIs
            patterns.append(MaskPattern(
                name="doi",
                pattern=re.compile(r'(?:doi:|DOI:|https?://doi\.org/)10\.\d{4,}/\S+'),
                priority=65
            ))
            
            # arXiv IDs
            patterns.append(MaskPattern(
                name="arxiv",
                pattern=re.compile(r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?'),
                priority=65
            ))
            
        if self.config.mask_emails:
            patterns.append(MaskPattern(
                name="email",
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                priority=55
            ))
            
        if self.config.mask_references:
            # Academic citations [1], [1,2,3], [1-5]
            patterns.append(MaskPattern(
                name="citation_bracket",
                pattern=re.compile(r'\[\d+(?:[,-]\d+)*\]'),
                priority=50
            ))
            
            # Author-year citations (Smith, 2020), (Smith et al., 2020)
            patterns.append(MaskPattern(
                name="citation_author",
                pattern=re.compile(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)'),
                priority=50
            ))
            
        if self.config.mask_numbers:
            # Scientific notation
            patterns.append(MaskPattern(
                name="scientific_notation",
                pattern=re.compile(r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b'),
                priority=40
            ))
            
            # Numbers with units
            patterns.append(MaskPattern(
                name="number_with_unit",
                pattern=re.compile(
                    r'\b\d+(?:\.\d+)?\s*(?:nm|μm|mm|cm|m|km|mg|g|kg|'
                    r'ms|s|min|h|Hz|kHz|MHz|GHz|K|°C|°F|%)\b'
                ),
                priority=35
            ))
            
        # Sort by priority (highest first)
        return sorted(patterns, key=lambda p: p.priority, reverse=True)
    
    def mask_block(self, block: Block) -> Block:
        """
        Apply masking to a single block.
        
        Returns block with masked_text and masks populated.
        """
        if not block.source_text or not block.is_translatable:
            return block
            
        masked_text = block.source_text
        masks = []
        
        # Track what's already masked to avoid overlaps
        masked_ranges: List[Tuple[int, int]] = []
        
        for pattern_def in self.patterns:
            # Find all matches
            matches = list(pattern_def.pattern.finditer(masked_text))
            
            for match in matches:
                start, end = match.span()
                
                # Check if this range overlaps with already masked content
                if any(start < m_end and end > m_start 
                      for m_start, m_end in masked_ranges):
                    continue  # Skip overlapping matches
                    
                # Generate placeholder
                placeholder = self._generate_placeholder(pattern_def.name)
                
                # Store mask info
                mask_info = MaskInfo(
                    original=match.group(0),
                    placeholder=placeholder,
                    mask_type=pattern_def.name,
                    preserve_formatting=pattern_def.preserve_formatting
                )
                masks.append(mask_info)
                self.mask_registry[placeholder] = mask_info
                
                # Replace in text (adjust for previous replacements)
                offset = len(placeholder) - len(match.group(0))
                masked_text = (
                    masked_text[:start] + 
                    placeholder + 
                    masked_text[end:]
                )
                
                # Update masked ranges
                masked_ranges.append((start, start + len(placeholder)))
                
                # Adjust future ranges for the length change
                masked_ranges = [
                    (s if s < start else s + offset,
                     e if e <= start else e + offset)
                    for s, e in masked_ranges
                ]
                
        block.masked_text = masked_text
        block.masks = masks
        
        logger.debug(f"Masked {len(masks)} patterns in block {block.block_id}")
        
        return block
    
    def unmask_block(self, block: Block, validate: bool = True) -> Block:
        """
        Restore masked content in a block.
        
        Args:
            block: Block with translated_text to unmask
            validate: Whether to validate all masks were preserved
            
        Returns:
            Block with unmasked translated_text
        """
        if not block.translated_text or not block.masks:
            return block
            
        unmasked_text = block.translated_text
        restored_count = 0
        missing_masks = []
        
        # Sort masks by placeholder length (longest first) to avoid partial replacements
        sorted_masks = sorted(block.masks, key=lambda m: len(m.placeholder), reverse=True)
        
        for mask in sorted_masks:
            if mask.placeholder in unmasked_text:
                unmasked_text = unmasked_text.replace(mask.placeholder, mask.original)
                restored_count += 1
            else:
                missing_masks.append(mask.placeholder)
                
        if validate and self.config.strict_validation and missing_masks:
            error_msg = f"Block {block.block_id}: Missing masks: {missing_masks}"
            self.validation_errors.append(error_msg)
            if self.config.log_validation_errors:
                logger.error(error_msg)
                
        # Update block
        block.translated_text = unmasked_text
        
        # Add validation stats to metadata
        if block.metadata:
            block.metadata.glossary_terms_used.add(f"masks_restored:{restored_count}")
            if missing_masks:
                block.metadata.glossary_terms_used.add(f"masks_missing:{len(missing_masks)}")
                
        logger.debug(f"Restored {restored_count}/{len(block.masks)} masks in block {block.block_id}")
        
        return block
    
    def mask_document(self, document: Document) -> Document:
        """Apply masking to entire document."""
        for segment in document.segments:
            for block in segment.blocks:
                self.mask_block(block)
        
        document.stats['total_masks'] = sum(len(b.masks) for b in document.all_blocks)
        return document
    
    def unmask_document(self, document: Document, validate: bool = True) -> Document:
        """Restore masked content in entire document."""
        self.validation_errors.clear()
        
        for segment in document.segments:
            for block in segment.blocks:
                self.unmask_block(block, validate=validate)
                
        if self.validation_errors:
            document.stats['mask_validation_errors'] = len(self.validation_errors)
            logger.warning(f"Document {document.document_id}: {len(self.validation_errors)} validation errors")
            
        return document
    
    def validate_masks(self, original_text: str, translated_text: str, 
                      masks: List[MaskInfo]) -> Tuple[bool, List[str]]:
        """
        Validate that all masks are preserved in translation.
        
        Returns:
            Tuple of (is_valid, list_of_missing_placeholders)
        """
        missing = []
        for mask in masks:
            if mask.placeholder not in translated_text:
                missing.append(mask.placeholder)
                
        return len(missing) == 0, missing
    
    def _generate_placeholder(self, mask_type: str) -> str:
        """Generate unique placeholder token."""
        self.mask_counter[mask_type] += 1
        count = self.mask_counter[mask_type]
        # Use distinctive format that won't appear naturally
        return f"<<{mask_type.upper()}_{count:04d}>>"
    
    def get_statistics(self) -> Dict[str, any]:
        """Get masking statistics."""
        return {
            'total_masks': len(self.mask_registry),
            'masks_by_type': dict(self.mask_counter),
            'validation_errors': len(self.validation_errors),
            'patterns_configured': len(self.patterns)
        }
    
    def reset(self):
        """Reset the engine for a new document."""
        self.mask_registry.clear()
        self.mask_counter.clear()
        self.validation_errors.clear()

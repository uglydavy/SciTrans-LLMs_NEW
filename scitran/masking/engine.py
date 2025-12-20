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
    mask_apostrophes_in_latex: bool = True  # Treat apostrophes inside math as part of math
    mask_custom_macros: bool = True  # Broader macro handling
    
    # Placeholder style (backend-aware)
    placeholder_style: str = "angle"  # "angle" (<<TYPE_0001>>) or "alnum" (SCITRANS_TYPE_0001_SCITRANS)
    placeholder_prefix: str = "SCITRANS"  # Prefix for alnum style
    placeholder_suffix: str = "SCITRANS"  # Suffix for alnum style
    placeholder_case: str = "upper"  # "upper" or "lower"
    
    # Validation settings
    strict_validation: bool = True  # Fail if masks can't be restored
    log_validation_errors: bool = True
    tolerant_unmasking: bool = True  # Try variant matching if exact match fails
    
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
            # Protect apostrophes inside math to avoid mangling
            if self.config.mask_apostrophes_in_latex:
                patterns.append(MaskPattern(
                    name="latex_apostrophe",
                    pattern=re.compile(r'(\$[^$]*?[\'][^$]*?\$|\\\(.*?\\\))', re.DOTALL),
                    priority=105,
                    preserve_formatting=True
                ))
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
            # LaTeX parenthetical inline \( ... \)
            patterns.append(MaskPattern(
                name="latex_inline_paren",
                pattern=re.compile(r'\\\(.*?\\\)', re.DOTALL),
                priority=90
            ))
            # LaTeX text/roman/bold/blackboard operators inside math
            patterns.append(MaskPattern(
                name="latex_text_ops",
                pattern=re.compile(r'\\(?:text|mathrm|mathbf|mathbb|mathcal|mathfrak|mathsf|operatorname)\{.*?\}', re.DOTALL),
                priority=92  # above inline math to preserve operators
            ))
            if self.config.mask_custom_macros:
                # Custom macro definitions (multi-line)
                patterns.append(MaskPattern(
                    name="latex_newcommand",
                    pattern=re.compile(r'\\newcommand\{\\[A-Za-z]+\}\s*\{.*?\}', re.DOTALL),
                    priority=87
                ))
                patterns.append(MaskPattern(
                    name="latex_newcommand_opt",
                    pattern=re.compile(r'\\newcommand\{\\[A-Za-z]+\}\s*\[[^\]]+\]\s*\{.*?\}', re.DOTALL),
                    priority=87
                ))
                # DeclareMathOperator
                patterns.append(MaskPattern(
                    name="latex_declare_operator",
                    pattern=re.compile(r'\\DeclareMathOperator\*?\{\\[A-Za-z]+\}\{.*?\}', re.DOTALL),
                    priority=87
                ))
                # Providecommand
                patterns.append(MaskPattern(
                    name="latex_provide_command",
                    pattern=re.compile(r'\\providecommand\{\\[A-Za-z]+\}\s*\{.*?\}', re.DOTALL),
                    priority=86
                ))
            # Common escaped braces/percents inside math
            patterns.append(MaskPattern(
                name="latex_escaped_brace",
                pattern=re.compile(r'\\[{}%]'),
                priority=86
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
        
        # First, find all matches across all patterns on the original text
        all_matches: List[Tuple[int, int, int, str, str, bool]] = []  # (start, end, priority, original, pattern_name, preserve_fmt)
        
        for pattern_def in self.patterns:
            matches = list(pattern_def.pattern.finditer(block.source_text))
            for match in matches:
                start, end = match.span()
                all_matches.append((
                    start, 
                    end, 
                    pattern_def.priority,
                    match.group(0),
                    pattern_def.name,
                    pattern_def.preserve_formatting
                ))
        
        # Sort by start position, then priority (higher first)
        all_matches.sort(key=lambda x: (x[0], -x[2]))
        
        # Remove overlapping matches (keep higher priority ones - they were added first due to pattern order)
        non_overlapping = []
        for match in all_matches:
            start, end, priority = match[0], match[1], match[2]
            overlaps = False
            for idx, accepted in enumerate(non_overlapping):
                a_start, a_end, a_pri = accepted[0], accepted[1], accepted[2]
                if start < a_end and end > a_start:
                    # Allow nested masks (one fully contained in the other)
                    if (start >= a_start and end <= a_end) or (a_start >= start and a_end <= end):
                        overlaps = False
                        break
                    overlaps = True
                    if priority > a_pri:
                        non_overlapping[idx] = match  # replace with higher priority
                    break
            if not overlaps:
                non_overlapping.append(match)
        
        # Sort by position in reverse order for safe replacement
        non_overlapping.sort(key=lambda x: x[0], reverse=True)
        
        # Now apply replacements from end to start
        masked_text = block.source_text
        masks = []
        
        for start, end, priority, original, pattern_name, preserve_fmt in non_overlapping:
            # Generate placeholder
            placeholder = self._generate_placeholder(pattern_name)
            
            # Store mask info
            mask_info = MaskInfo(
                original=original,
                placeholder=placeholder,
                mask_type=pattern_name,
                preserve_formatting=preserve_fmt
            )
            masks.append(mask_info)
            self.mask_registry[placeholder] = mask_info
            
            # Replace in text (from end to start, so positions stay valid)
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
        
        # Reverse masks to maintain original order
        masks.reverse()
        
        block.masked_text = masked_text
        block.masks = masks
        
        logger.debug(f"Masked {len(masks)} patterns in block {block.block_id}")
        
        return block
    
    def find_placeholder_variants(self, placeholder: str) -> List[str]:
        """Generate regex patterns for tolerant placeholder matching.
        
        Args:
            placeholder: Original placeholder (e.g., "<<MATH_0001>>")
        
        Returns:
            List of regex patterns that match common mutations
        """
        patterns = []
        
        # Extract parts from placeholder
        if placeholder.startswith("<<") and placeholder.endswith(">>"):
            # Angle style: <<TYPE_0001>>
            inner = placeholder[2:-2]  # TYPE_0001
            parts = inner.split("_")
            if len(parts) >= 2:
                mask_type, mask_id = parts[0], parts[1]
                
                # Pattern 1: Exact match
                patterns.append(re.escape(placeholder))
                
                # Pattern 2: With arbitrary spaces
                patterns.append(r"<<\s*" + re.escape(mask_type) + r"\s*_\s*" + re.escape(mask_id) + r"\s*>>")
                
                # Pattern 3: Guillemets variants
                patterns.append(r"«\s*" + re.escape(mask_type) + r"\s*_\s*" + re.escape(mask_id) + r"\s*»")
                patterns.append(r"‹\s*" + re.escape(mask_type) + r"\s*_\s*" + re.escape(mask_id) + r"\s*›")
                
                # Pattern 4: Case-insensitive
                patterns.append(r"<<\s*" + mask_type + r"\s*_\s*" + mask_id + r"\s*>>")
                
        elif self.config.placeholder_prefix in placeholder and self.config.placeholder_suffix in placeholder:
            # Alnum style: SCITRANS_TYPE_0001_SCITRANS
            parts = placeholder.split("_")
            if len(parts) >= 4:
                prefix, mask_type, mask_id, suffix = parts[0], parts[1], parts[2], parts[3]
                
                # Pattern 1: Exact
                patterns.append(re.escape(placeholder))
                
                # Pattern 2: With spaces
                patterns.append(re.escape(prefix) + r"\s*_\s*" + re.escape(mask_type) + r"\s*_\s*" + re.escape(mask_id) + r"\s*_\s*" + re.escape(suffix))
                
                # Pattern 3: Case variants
                patterns.append(prefix + r"_" + mask_type + r"_" + mask_id + r"_" + suffix)
        
        return patterns
    
    def unmask_block(self, block: Block, validate: bool = True) -> Block:
        """
        Restore masked content in a block with tolerant matching.
        
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
            # Try exact match first
            if mask.placeholder in unmasked_text:
                unmasked_text = unmasked_text.replace(mask.placeholder, mask.original)
                restored_count += 1
            elif self.config.tolerant_unmasking:
                # Try variant matching
                found = False
                for pattern in self.find_placeholder_variants(mask.placeholder):
                    matches = list(re.finditer(pattern, unmasked_text, re.IGNORECASE))
                    if matches:
                        # Replace all matches of this variant
                        for match in reversed(matches):  # Reverse to preserve positions
                            unmasked_text = unmasked_text[:match.start()] + mask.original + unmasked_text[match.end():]
                        restored_count += 1
                        found = True
                        logger.debug(f"Restored {mask.placeholder} using variant pattern: {pattern}")
                        break
                
                if not found:
                    missing_masks.append(mask.placeholder)
            else:
                missing_masks.append(mask.placeholder)
                
        # Store restoration metadata in block
        if not hasattr(block, 'metadata') or block.metadata is None:
            from scitran.core.models import TranslationMetadata
            from datetime import datetime
            block.metadata = TranslationMetadata(
                backend="unknown",
                timestamp=datetime.now(),
                duration=0.0
            )
        
        block.metadata.restored_masks = restored_count
        block.metadata.missing_placeholders = missing_masks
                
        if validate and self.config.strict_validation and missing_masks:
            error_msg = f"Block {block.block_id}: Missing masks: {missing_masks}"
            self.validation_errors.append(error_msg)
            if self.config.log_validation_errors:
                logger.error(error_msg)
                
        # Update block
        block.translated_text = unmasked_text
        
        # Add validation stats to metadata (if TranslationMetadata object)
        if block.metadata and hasattr(block.metadata, 'glossary_terms_used'):
            try:
                block.metadata.glossary_terms_used.add(f"masks_restored:{restored_count}")
                if missing_masks:
                    block.metadata.glossary_terms_used.add(f"masks_missing:{len(missing_masks)}")
            except (AttributeError, TypeError):
                pass  # metadata is a dict or doesn't support this
                
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
        """Generate unique placeholder token based on configured style."""
        self.mask_counter[mask_type] += 1
        count = self.mask_counter[mask_type]
        
        mask_type_upper = mask_type.upper() if self.config.placeholder_case == "upper" else mask_type.lower()
        
        if self.config.placeholder_style == "alnum":
            # Alphanumeric only - safer for free backends
            # Format: SCITRANS_TYPE_0001_SCITRANS
            prefix = self.config.placeholder_prefix.upper() if self.config.placeholder_case == "upper" else self.config.placeholder_prefix.lower()
            suffix = self.config.placeholder_suffix.upper() if self.config.placeholder_case == "upper" else self.config.placeholder_suffix.lower()
            return f"{prefix}_{mask_type_upper}_{count:04d}_{suffix}"
        else:
            # Angle bracket style (default) - works well with LLM backends
            # Format: <<TYPE_0001>>
            return f"<<{mask_type_upper}_{count:04d}>>"
    
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

    # Convenience for unit tests / standalone usage
    def apply_text(self, text: str) -> str:
        """Apply masking to raw text (non-block)."""
        from scitran.core.models import Block, BlockType, BoundingBox
        tmp_block = Block(
            block_id="tmp",
            source_text=text,
            block_type=BlockType.PARAGRAPH,
            bbox=BoundingBox(0, 0, 1, 1, page=0),
            masks=[],
        )
        masked_block = self.mask_block(tmp_block)
        return masked_block.masked_text or masked_block.source_text

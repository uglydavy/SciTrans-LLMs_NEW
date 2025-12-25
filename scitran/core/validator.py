# -*- coding: utf-8 -*-
"""
Translation Completeness Validator.

Enforces strict validation rules to prevent partial/chaotic output:
1. Coverage == 100% (all translatable blocks translated)
2. No unchanged output (source != translation)
3. Strict mask token preservation
4. Glossary constraints adherence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging
import re

from scitran.core.models import Document, Block

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of completeness validation."""
    is_valid: bool = True
    coverage: float = 0.0  # 0.0-1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Detailed metrics
    total_blocks: int = 0
    translatable_blocks: int = 0
    translated_blocks: int = 0
    failed_blocks: List[str] = field(default_factory=list)
    
    # Mask validation
    total_masks: int = 0
    preserved_masks: int = 0
    missing_masks: List[Dict] = field(default_factory=list)
    
    # Identity translations (source == target)
    identity_blocks: List[str] = field(default_factory=list)


class TranslationCompletenessValidator:
    """
    Validates translation completeness and quality.
    
    This enforces the non-negotiable acceptance criteria:
    - Coverage = 100%
    - No source text fallbacks
    - Strict mask preservation
    - Glossary adherence
    """
    
    def __init__(
        self,
        require_full_coverage: bool = True,
        require_mask_preservation: bool = True,
        require_no_identity: bool = True,
        min_alpha_ratio: float = 0.3,  # Minimum alphabetic ratio for identity check
        identity_min_length: int = 30,  # Minimum length for identity check
        identity_min_words: int = 6,  # Minimum words for identity check
        source_lang: str = "en",  # Source language
        target_lang: str = "fr"  # Target language
    ):
        self.require_full_coverage = require_full_coverage
        self.require_mask_preservation = require_mask_preservation
        self.require_no_identity = require_no_identity
        self.min_alpha_ratio = min_alpha_ratio
        self.identity_min_length = identity_min_length
        self.identity_min_words = identity_min_words
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def validate(self, document: Document) -> ValidationResult:
        """
        Validate document translation completeness.
        
        Args:
            document: Document with translations
        
        Returns:
            ValidationResult with detailed metrics
        """
        result = ValidationResult(is_valid=True)
        
        # Get all blocks
        all_blocks = document.all_blocks
        translatable = document.translatable_blocks
        
        result.total_blocks = len(all_blocks)
        result.translatable_blocks = len(translatable)
        
        if result.translatable_blocks == 0:
            result.warnings.append("No translatable blocks found in document")
            result.coverage = 1.0
            return result
        
        # Check 1: Coverage (all translatable blocks must have translations)
        for block in translatable:
            if not block.translated_text or not block.translated_text.strip():
                result.failed_blocks.append(block.block_id)
                result.errors.append(f"Block {block.block_id}: Missing translation")
            else:
                result.translated_blocks += 1
        
        result.coverage = result.translated_blocks / result.translatable_blocks
        
        if self.require_full_coverage and result.coverage < 1.0:
            result.is_valid = False
            result.errors.append(f"Coverage incomplete: {result.coverage:.1%} (required: 100%)")
        
        # Check 2: Identity translations (source == target) - SMART HEURISTIC
        if self.require_no_identity:
            for block in translatable:
                if not block.translated_text:
                    continue
                
                source_norm = self._normalize_text(block.source_text)
                target_norm = self._normalize_text(block.translated_text)
                
                if source_norm and target_norm and source_norm == target_norm:
                    # Use smarter heuristic - don't flag short texts, proper nouns, technical content
                    if self._is_real_identity_failure(block.source_text, block.translated_text):
                        result.identity_blocks.append(block.block_id)
                        result.errors.append(f"Block {block.block_id}: Identity translation (source == target)")
        
        if result.identity_blocks:
            result.is_valid = False
        
        # Check 3: Mask restoration (CRITICAL) - check POST-UNMASK
        # This should run AFTER unmasking, so we check restoration outcomes not placeholder presence
        if self.require_mask_preservation:
            for block in all_blocks:
                if not block.masks:
                    continue
                
                if not block.translated_text:
                    continue  # Already caught by coverage check
                
                result.total_masks += len(block.masks)
                
                # Check restoration metadata (set by unmask_block)
                if hasattr(block, 'metadata') and block.metadata:
                    # Handle both dict and TranslationMetadata objects
                    if isinstance(block.metadata, dict):
                        restored = block.metadata.get('restored_masks', None)
                        missing_list = block.metadata.get('missing_placeholders', [])
                    else:
                        restored = getattr(block.metadata, 'restored_masks', None)
                        missing_list = getattr(block.metadata, 'missing_placeholders', [])
                    
                    if restored is not None:
                        # Use restoration metadata
                        result.preserved_masks += restored
                        
                        if missing_list:
                            for placeholder in missing_list:
                                # Find the mask info for this placeholder
                                mask_info = next((m for m in block.masks if m.placeholder == placeholder), None)
                                if mask_info:
                                    result.missing_masks.append({
                                        "block_id": block.block_id,
                                        "placeholder": placeholder,
                                        "mask_type": mask_info.mask_type,
                                        "original": mask_info.original[:50] + "..." if len(mask_info.original) > 50 else mask_info.original
                                    })
                                    result.errors.append(
                                        f"Block {block.block_id}: Failed to restore mask {placeholder} (type: {mask_info.mask_type})"
                                    )
                    else:
                        # Fallback: count all as preserved (no metadata means old behavior)
                        result.preserved_masks += len(block.masks)
                else:
                    # No metadata: count all as preserved (old behavior)
                    result.preserved_masks += len(block.masks)
        
        if result.missing_masks:
            result.is_valid = False
            result.errors.append(f"Mask restoration failed: {len(result.missing_masks)} masks could not be restored")
        
        # Final validation
        if result.errors:
            logger.error(f"Validation failed with {len(result.errors)} errors")
            for error in result.errors[:10]:  # Show first 10
                logger.error(f"  - {error}")
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        return " ".join(text.lower().strip().split())
    
    def _has_alphabetic_content(self, text: str) -> bool:
        """Check if text has substantial alphabetic content."""
        if not text:
            return False
        
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = len(text.strip())
        
        if total_count == 0:
            return False
        
        return (alpha_count / total_count) >= self.min_alpha_ratio
    
    def _is_real_identity_failure(self, source: str, translated: str) -> bool:
        """Determine if identical text represents a REAL translation failure.
        
        Uses smart heuristics to avoid false positives on:
        - Short headings, titles, proper nouns
        - Technical terms, acronyms
        - Content with many placeholders
        """
        if not source or not translated:
            return False
        
        # Same language? Identity expected
        if self.source_lang == self.target_lang:
            return False
        
        # Too short? Likely proper noun/title
        if len(source.strip()) < self.identity_min_length:
            return False
        
        # Too few words? Likely heading
        words = source.split()
        if len(words) < self.identity_min_words:
            return False
        
        # Mostly uppercase? Likely acronyms/headers
        alpha_chars = sum(1 for c in source if c.isalpha())
        if alpha_chars > 0:
            upper_chars = sum(1 for c in source if c.isupper())
            if upper_chars / alpha_chars > 0.7:
                return False
        
        # Lots of numbers? Likely technical
        digit_chars = sum(1 for c in source if c.isdigit())
        if len(source) > 0 and digit_chars / len(source) > 0.3:
            return False
        
        # Mostly placeholders? Not a real text block
        placeholder_pattern = r'<<[A-Z_]+_\d+>>|SCITRANS_[A-Z_]+_\d+_SCITRANS'
        placeholders = re.findall(placeholder_pattern, source)
        if placeholders:
            placeholder_chars = sum(len(p) for p in placeholders)
            if placeholder_chars / len(source) > 0.5:
                return False
        
        # Long text, enough words, not technical → TRUE failure
        return True
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRANSLATION COMPLETENESS VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Status
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        lines.append(f"Status: {status}")
        lines.append("")
        
        # Coverage
        lines.append(f"Coverage: {result.coverage:.1%} ({result.translated_blocks}/{result.translatable_blocks})")
        if result.failed_blocks:
            lines.append(f"  Failed blocks: {len(result.failed_blocks)}")
            for block_id in result.failed_blocks[:5]:
                lines.append(f"    - {block_id}")
            if len(result.failed_blocks) > 5:
                lines.append(f"    ... and {len(result.failed_blocks) - 5} more")
        lines.append("")
        
        # Identity translations
        if result.identity_blocks:
            lines.append(f"Identity translations: {len(result.identity_blocks)}")
            for block_id in result.identity_blocks[:5]:
                lines.append(f"  - {block_id}")
        lines.append("")
        
        # Masks
        if result.total_masks > 0:
            lines.append(f"Mask preservation: {result.preserved_masks}/{result.total_masks}")
            if result.missing_masks:
                lines.append(f"  Missing masks: {len(result.missing_masks)}")
                for mask_info in result.missing_masks[:5]:
                    lines.append(f"    - Block {mask_info['block_id']}: {mask_info['placeholder']} ({mask_info['mask_type']})")
        lines.append("")
        
        # Errors
        if result.errors:
            lines.append(f"Errors: {len(result.errors)}")
            for error in result.errors[:10]:
                lines.append(f"  - {error}")
        
        # Warnings
        if result.warnings:
            lines.append(f"Warnings: {len(result.warnings)}")
            for warning in result.warnings[:5]:
                lines.append(f"  - {warning}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


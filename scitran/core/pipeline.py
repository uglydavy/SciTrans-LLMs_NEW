"""
Main translation pipeline for SciTrans-LLMs NEW.

This module orchestrates the complete translation workflow, integrating
all three innovations into a cohesive system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import time
import logging
from datetime import datetime
import json
import os
import unicodedata

from scitran.core.models import (
    Document, Block, Segment, TranslationResult,
    TranslationMetadata, BlockType
)
from scitran.masking.engine import MaskingEngine, MaskingConfig
from scitran.translation.prompts import PromptOptimizer, PromptLibrary
from scitran.scoring.reranker import AdvancedReranker, ScoringStrategy

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete configuration for the translation pipeline."""
    
    # Language settings
    source_lang: str = "en"
    target_lang: str = "fr"
    
    # Translation backend
    backend: str = "openai"  # openai, anthropic, deepseek, ollama, free
    model_name: Optional[str] = None  # Specific model to use
    api_key: Optional[str] = None
    
    # Innovation #1: Masking configuration
    enable_masking: bool = True
    masking_config: MaskingConfig = field(default_factory=MaskingConfig)
    validate_mask_restoration: bool = True
    
    # Innovation #2: Document context
    enable_context: bool = True
    context_window_size: int = 5
    use_extended_context: bool = False  # Use context from entire document
    
    # Innovation #3: Layout preservation
    preserve_layout: bool = True
    layout_detection_method: str = "yolo"  # yolo, heuristic, hybrid
    
    # Glossary settings (SPRINT 3: Enhanced)
    enable_glossary: bool = True
    glossary_path: Optional[Path] = None
    glossary_domains: List[str] = field(default_factory=lambda: ["ml"])  # Domains to load
    domain: str = "scientific"  # Deprecated: use glossary_domains instead
    glossary_manager: Optional[Any] = None  # Pre-configured glossary manager
    glossary_strict: bool = False  # Fail if glossary violated
    
    # Quality settings
    num_candidates: int = 3  # Number of candidates to generate
    enable_reranking: bool = True
    reranking_strategy: ScoringStrategy = ScoringStrategy.HYBRID
    quality_threshold: float = 0.7  # Minimum acceptable quality score
    
    # SPRINT 1: Translation coverage guarantee
    strict_mode: bool = True  # Fail loudly if any blocks untranslated
    max_translation_retries: int = 3  # Retry failed blocks this many times
    retry_backoff_factor: float = 2.0  # Exponential backoff multiplier
    enable_fallback_backend: bool = True  # Escalate to stronger backend on failure
    fallback_backend: str = "openai"  # Backend to use for failed blocks
    detect_identity_translation: bool = True  # Treat source==output as failure
    
    # Prompt settings
    prompt_template: str = "scientific_expert"
    optimize_prompts: bool = True
    prompt_optimization_rounds: int = 3
    
    # SPRINT 4: Document-level refinement
    enable_refinement: bool = False  # Enable document-level refinement pass
    refinement_backend: Optional[str] = None  # Backend for refinement (defaults to main backend)
    refinement_prompt: str = "coherence"  # Refinement focus: coherence, style, terminology
    validate_refinement_constraints: bool = True  # Ensure placeholders/glossary preserved
    
    # Performance settings
    batch_size: int = 10  # Blocks to translate in parallel
    cache_translations: bool = True
    cache_dir: Optional[Path] = None
    max_retries: int = 3
    timeout: int = 30  # Seconds per block
    
    # Output settings
    output_format: str = "pdf"  # pdf, docx, txt, json
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # ABLATION FLAGS (SPRINT 4): For thesis experiments
    # Set all to False to disable innovations and create baseline
    ablation_disable_masking: bool = False       # Disable Innovation #1: Masking
    ablation_disable_glossary: bool = False      # Disable glossary enforcement  
    ablation_disable_context: bool = False       # Disable Innovation #2: Context
    ablation_disable_reranking: bool = False     # Disable multi-candidate reranking
    ablation_disable_refinement: bool = False    # Disable Innovation #3: Refinement
    ablation_disable_coverage_guarantee: bool = False  # Disable retry/fallback
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        if self.num_candidates < 1:
            issues.append("num_candidates must be at least 1")
            
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            issues.append("quality_threshold must be between 0 and 1")
            
        if self.context_window_size < 0:
            issues.append("context_window_size must be non-negative")
            
        if self.backend == "openai" and not self.api_key:
            issues.append("OpenAI backend requires api_key")
            
        return issues


class TranslationPipeline:
    """
    Main translation pipeline implementing all three innovations.
    
    This is the core orchestrator that:
    1. Applies masking to protect content
    2. Maintains document-level context
    3. Generates and reranks candidates
    4. Validates quality
    5. Preserves layout
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, progress_callback: Optional[Callable[[float, str], None]] = None):
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            raise ValueError(f"Configuration issues: {', '.join(issues)}")
            
        # Initialize components
        self.masking_engine = MaskingEngine(self.config.masking_config) if self.config.enable_masking else None
        self.prompt_optimizer = PromptOptimizer() if self.config.optimize_prompts else None
        self.reranker = AdvancedReranker(strategy=self.config.reranking_strategy) if self.config.enable_reranking else None
        
        # Translation backend (will be initialized later)
        self.translator = None
        
        # Translation cache for sequential mode
        self._translation_cache: Optional[Any] = None
        if self.config.cache_translations:
            try:
                from scitran.utils.fast_translator import PersistentCache
                cache_dir = str(self.config.cache_dir) if self.config.cache_dir else ".cache/translations"
                self._translation_cache = PersistentCache(cache_dir)
            except ImportError:
                logger.warning("Cache not available, translations won't be cached")
        
        # Glossary (will be loaded later)
        self.glossary: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'blocks_processed': 0,
            'blocks_succeeded': 0,
            'blocks_failed': 0,
            'total_candidates': 0,
            'reranking_improvements': 0,
            'mask_violations': 0,
            'quality_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Progress callback
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        
    def translate_document(self, 
                          document: Document,
                          progress_callback: Optional[Callable[[float, str], None]] = None) -> TranslationResult:
        """
        Translate a complete document with all innovations applied.
        
        Args:
            document: Document to translate
            progress_callback: Optional callback for progress updates
            
        Returns:
            TranslationResult with translated document and metrics
        """
        start_time = time.time()
        self.progress_callback = progress_callback
        # Keep a reference for batch heuristics
        self.document = document
        self._dbg_log("H0", "pipeline:start", "translate_document start", {"pages": document.stats.get("num_pages", 0)})
        
        # Initialize result
        result = TranslationResult(
            document=document,
            backend_used=self.config.backend
        )
        
        try:
            # Setup
            self._setup_translation(document)
            
            # Check if masking should be disabled for this backend
            # Free services can't preserve mask tokens, causing content loss
            # SPRINT 4: Also check ablation flag
            use_masking = self.config.enable_masking and not self.config.ablation_disable_masking
            if use_masking and self._should_disable_masking():
                self._report_progress(0.05, "Note: Masking disabled for free backend (can't preserve tokens)")
                use_masking = False
            
            # Phase 1: Apply masking (Innovation #1)
            if use_masking:
                self._report_progress(0.1, "Applying content masking...")
                self.masking_engine.reset()
                document = self.masking_engine.mask_document(document)
                result.masks_applied = document.stats.get('total_masks', 0)
            
            # Phase 2: Translate blocks with context (Innovation #2)
            self._report_progress(0.2, "Translating document...")
            self._translate_all_blocks(document)
            
            # Phase 2.5: SPRINT 1 - Translation coverage guarantee
            # SPRINT 4: Skip if ablation disabled
            if not self.config.ablation_disable_coverage_guarantee:
                self._report_progress(0.7, "Ensuring complete coverage...")
                self._ensure_translation_coverage(document, result)
            
            # Phase 2.75: SPRINT 4 - Document-level refinement (if enabled)
            if self.config.enable_refinement and not self.config.ablation_disable_refinement:
                self._report_progress(0.75, "Refining translation coherence...")
                self._refine_document_translation(document, result)
            
            # Phase 3: Restore masks with validation
            if use_masking:
                self._report_progress(0.8, "Restoring masked content...")
                document = self.masking_engine.unmask_document(
                    document, 
                    validate=self.config.validate_mask_restoration
                )
                result.masks_restored = result.masks_applied - len(self.masking_engine.validation_errors)
                
            # Phase 4: Final validation
            self._report_progress(0.9, "Validating translation quality...")
            quality_metrics = self._validate_translation(document)
            result.bleu_score = quality_metrics.get('bleu')
            result.chrf_score = quality_metrics.get('chrf')
            result.glossary_adherence = quality_metrics.get('glossary_adherence')
            result.layout_preservation = quality_metrics.get('layout_preservation')
            
            # Update statistics
            document.update_stats()
            result.blocks_translated = self.stats['blocks_succeeded']
            result.blocks_failed = self.stats['blocks_failed']
            result.success = result.blocks_failed == 0
            
            self._report_progress(1.0, "Translation complete!")
            self._dbg_log(
                "H5",
                "pipeline:complete",
                "translate_document complete",
                {
                    "duration_sec": time.time() - start_time,
                    "blocks_translated": result.blocks_translated,
                    "blocks_failed": result.blocks_failed,
                    "mask_violations": self.stats.get('mask_violations', 0),
                    "cache_hits": self.stats.get('cache_hits', 0),
                    "cache_misses": self.stats.get('cache_misses', 0),
                },
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.success = False
            result.error = str(e)
            
        result.duration = time.time() - start_time
        
        return result
    
    def _setup_translation(self, document: Document):
        """Initialize translation components."""
        # SPRINT 3: Load glossary using GlossaryManager
        # SPRINT 4: Skip if ablation disabled
        if self.config.enable_glossary and not self.config.ablation_disable_glossary:
            self.glossary_manager = self._setup_glossary_manager()
            # Keep legacy glossary dict for compatibility
            self.glossary = self.glossary_manager.to_dict() if self.glossary_manager else {}
            self._dbg_log("H1", "pipeline:glossary_loaded", "glossary loaded", {
                "entries": len(self.glossary_manager) if self.glossary_manager else 0,
                "domains": list(self.glossary_manager.domains_loaded) if self.glossary_manager else []
            })
        else:
            self.glossary_manager = None
            self.glossary = {}
            
        # Initialize translator
        self.translator = self._create_translator()
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}
        
    def _translate_all_blocks(self, document: Document):
        """Translate all blocks in the document with batch processing."""
        total_blocks = len(document.translatable_blocks)
        
        # Check if batch translation is available
        use_batch = self._can_use_batch_translation(total_blocks)
        
        if use_batch and total_blocks > 1:
            self._translate_blocks_batch(document)
        else:
            self._translate_blocks_sequential(document)
    
    def _can_use_batch_translation(self, total_blocks: int = 0) -> bool:
        """Check if batch translation is available for current backend.
        
        Batch mode is faster because it:
        - Uses async concurrent requests
        - Has built-in caching
        - Deduplicates identical texts
        
        For large documents (>50 blocks), always use batch mode for speed,
        even with reranking disabled (reranking can be done post-batch if needed).
        """
        simple_backends = {'cascade', 'free', 'huggingface'}
        
        # Always use batch mode for free backends (they're fast enough)
        # For large documents, prioritize speed over reranking
        if self.config.backend.lower() in simple_backends:
            # Use batch mode if: no reranking OR large document (>50 blocks)
            return not self.config.enable_reranking or total_blocks > 50
        
        return False
    
    def _retry_failed_blocks(self, document: Document, block_ids: List[str]) -> None:
        """Retry translation for specific blocks that were missing from batch results."""
        if not block_ids:
            return
        
        # Map for quick lookup
        id_set = set(block_ids)
        for block in document.translatable_blocks:
            if block.block_id not in id_set:
                continue
            try:
                text = block.masked_text or block.source_text
                translation = self._call_translator(system_prompt=None, user_prompt=text)
                if translation:
                    block.translated_text = translation
                    self.stats['blocks_succeeded'] += 1
                    if self.stats.get('blocks_failed', 0) > 0:
                        self.stats['blocks_failed'] -= 1
                else:
                    self.stats['blocks_failed'] += 1
            except Exception as e:
                logger.error(f"Retry failed for block {block.block_id}: {e}")
                self.stats['blocks_failed'] += 1
    
    def _ensure_translation_coverage(self, document: Document, result: TranslationResult) -> None:
        """SPRINT 1: Ensure ALL translatable blocks have valid translations.
        
        This implements the translation coverage guarantee:
        1. Detect missing translations (None or empty)
        2. Detect identity translations (source == output)
        3. Retry with exponential backoff
        4. Fallback to stronger backend if retries fail
        5. Generate failure report in strict mode
        
        Args:
            document: Document with potentially incomplete translations
            result: TranslationResult to update with coverage metrics
        
        Raises:
            TranslationCoverageError: If strict mode and coverage incomplete
        """
        from scitran.core.exceptions import TranslationCoverageError
        
        # Phase 1: Detect missing/identity translations
        missing_blocks = self._detect_missing_translations(document)
        
        if not missing_blocks:
            logger.info("✓ Translation coverage: 100%")
            result.coverage = 1.0
            return
        
        initial_missing = len(missing_blocks)
        logger.warning(f"Translation coverage incomplete: {initial_missing} blocks missing/identity")
        
        # Phase 2: Retry with exponential backoff
        if self.config.max_translation_retries > 0:
            missing_blocks = self._retry_with_backoff(document, missing_blocks)
        
        # Phase 3: Fallback to stronger backend if still missing
        if missing_blocks and self.config.enable_fallback_backend:
            missing_blocks = self._fallback_translate(document, missing_blocks)
        
        # Phase 4: Calculate final coverage
        final_missing = len(missing_blocks)
        total_translatable = len(document.translatable_blocks)
        coverage = (total_translatable - final_missing) / total_translatable if total_translatable > 0 else 1.0
        result.coverage = coverage
        
        # Phase 5: Handle remaining failures
        if missing_blocks:
            failure_report = self._generate_failure_report(document, missing_blocks)
            result.failure_report = failure_report
            
            if self.config.strict_mode:
                logger.error(f"✗ Translation coverage: {coverage:.1%} - STRICT mode: failing")
                raise TranslationCoverageError(
                    f"Translation incomplete: {final_missing}/{total_translatable} blocks failed",
                    failure_report=failure_report
                )
            else:
                logger.warning(f"⚠ Translation coverage: {coverage:.1%} - proceeding with partial translation")
        else:
            logger.info(f"✓ Translation coverage: 100% (recovered {initial_missing} blocks)")
    
    def _detect_missing_translations(self, document: Document) -> List[Block]:
        """Detect blocks with missing or identity translations.
        
        Returns:
            List of blocks that need (re)translation
        """
        missing = []
        
        for block in document.translatable_blocks:
            # Check 1: No translation at all
            if not block.translated_text or block.translated_text.strip() == "":
                logger.debug(f"Block {block.block_id}: missing translation")
                missing.append(block)
                continue
            
            # Check 2: Identity translation detection (if enabled)
            if self.config.detect_identity_translation:
                source = self._normalize_text(block.source_text)
                target = self._normalize_text(block.translated_text)
                
                # Only flag as identity if substantial alphabetic content matches
                if source and target and source == target:
                    # Check if text has meaningful content (not just numbers/symbols)
                    if self._has_alphabetic_content(source):
                        logger.debug(f"Block {block.block_id}: identity translation detected")
                        missing.append(block)
                        continue
        
        return missing
    
    def _retry_with_backoff(self, document: Document, blocks: List[Block]) -> List[Block]:
        """Retry translation for failed blocks with exponential backoff.
        
        Args:
            document: Document containing blocks
            blocks: List of blocks to retry
        
        Returns:
            List of blocks that still failed after retries
        """
        import time
        
        remaining = blocks.copy()
        retry_delay = 1.0  # Initial delay in seconds
        
        for attempt in range(1, self.config.max_translation_retries + 1):
            if not remaining:
                break
            
            logger.info(f"Retry attempt {attempt}/{self.config.max_translation_retries} for {len(remaining)} blocks")
            
            # Wait before retry (exponential backoff)
            if attempt > 1:
                time.sleep(retry_delay)
                retry_delay *= self.config.retry_backoff_factor
            
            # Retry each block
            still_missing = []
            for block in remaining:
                try:
                    text = block.masked_text or block.source_text
                    translation = self._call_translator(system_prompt=None, user_prompt=text)
                    
                    if translation and translation.strip():
                        # Check not identity
                        if self.config.detect_identity_translation:
                            if self._normalize_text(translation) != self._normalize_text(block.source_text):
                                block.translated_text = translation
                                logger.debug(f"✓ Block {block.block_id} recovered on retry {attempt}")
                                continue
                        else:
                            block.translated_text = translation
                            logger.debug(f"✓ Block {block.block_id} recovered on retry {attempt}")
                            continue
                    
                    still_missing.append(block)
                    
                except Exception as e:
                    logger.debug(f"Retry {attempt} failed for block {block.block_id}: {e}")
                    still_missing.append(block)
            
            remaining = still_missing
        
        if remaining:
            logger.warning(f"Retries exhausted: {len(remaining)} blocks still missing")
        
        return remaining
    
    def _fallback_translate(self, document: Document, blocks: List[Block]) -> List[Block]:
        """Use fallback backend for failed blocks.
        
        Args:
            document: Document containing blocks
            blocks: List of blocks that failed with primary backend
        
        Returns:
            List of blocks that still failed after fallback
        """
        if not blocks:
            return []
        
        fallback = self.config.fallback_backend
        if fallback == self.config.backend:
            logger.warning(f"Fallback backend same as primary ({fallback}), skipping")
            return blocks
        
        logger.info(f"Escalating {len(blocks)} blocks to fallback backend: {fallback}")
        
        # Create fallback translator
        try:
            fallback_translator = self._create_translator(backend_override=fallback)
        except Exception as e:
            logger.error(f"Failed to create fallback translator: {e}")
            return blocks
        
        still_missing = []
        for block in blocks:
            try:
                text = block.masked_text or block.source_text
                # Use fallback translator
                request = self._build_translation_request(text)
                response = fallback_translator.translate_sync(request)
                
                if response.translations and response.translations[0]:
                    translation = response.translations[0]
                    # Validate not identity
                    if self._normalize_text(translation) != self._normalize_text(block.source_text):
                        block.translated_text = translation
                        block.metadata = block.metadata or TranslationMetadata()
                        block.metadata.backend_used = fallback
                        logger.debug(f"✓ Block {block.block_id} recovered via fallback")
                        continue
                
                still_missing.append(block)
                
            except Exception as e:
                logger.debug(f"Fallback translation failed for block {block.block_id}: {e}")
                still_missing.append(block)
        
        if still_missing:
            logger.warning(f"Fallback exhausted: {len(still_missing)} blocks still missing")
        
        return still_missing
    
    def _generate_failure_report(self, document: Document, failed_blocks: List[Block]) -> Dict[str, Any]:
        """Generate machine-readable failure report for missing translations.
        
        Args:
            document: Document with failures
            failed_blocks: List of blocks that couldn't be translated
        
        Returns:
            Failure report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "document_id": document.document_id,
            "total_blocks": len(document.translatable_blocks),
            "failed_count": len(failed_blocks),
            "failures": []
        }
        
        for block in failed_blocks:
            failure = {
                "block_id": block.block_id,
                "source_text": block.source_text[:100] + "..." if len(block.source_text) > 100 else block.source_text,
                "page": block.bbox.page if block.bbox else None,
                "bbox": {
                    "x0": block.bbox.x0,
                    "y0": block.bbox.y0,
                    "x1": block.bbox.x1,
                    "y1": block.bbox.y1,
                } if block.bbox else None,
                "block_type": block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type),
                "reason": "missing_translation" if not block.translated_text else "identity_translation"
            }
            report["failures"].append(failure)
        
        return report
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip whitespace)."""
        if not text:
            return ""
        return " ".join(text.lower().strip().split())
    
    def _has_alphabetic_content(self, text: str, min_alpha_ratio: float = 0.3) -> bool:
        """Check if text has substantial alphabetic content.
        
        Args:
            text: Text to check
            min_alpha_ratio: Minimum ratio of alphabetic chars (0-1)
        
        Returns:
            True if text has meaningful alphabetic content
        """
        if not text:
            return False
        
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = len(text.strip())
        
        if total_count == 0:
            return False
        
        return (alpha_count / total_count) >= min_alpha_ratio
    
    def _refine_document_translation(self, document: Document, result: TranslationResult) -> None:
        """SPRINT 4: Document-level refinement pass for coherence and consistency.
        
        This pass improves translation quality while preserving:
        - Masked placeholders (MATH, URL, CODE, etc.)
        - Glossary-enforced terms
        - Layout structure
        
        Args:
            document: Document with initial translations
            result: TranslationResult to update with refinement metrics
        
        Raises:
            ValueError: If refinement breaks constraints
        """
        if not self.config.enable_refinement:
            return
        
        logger.info("Starting document-level refinement pass")
        
        # Store original translations for rollback if needed
        original_translations = {
            block.block_id: block.translated_text 
            for block in document.translatable_blocks
        }
        
        refinement_stats = {
            'blocks_refined': 0,
            'blocks_failed': 0,
            'constraint_violations': 0
        }
        
        # Refine each segment for coherence
        for segment in document.segments:
            translatable = segment.translatable_blocks
            if not translatable:
                continue
            
            # Build context: all translations in segment
            segment_translations = [
                b.translated_text for b in translatable if b.translated_text
            ]
            
            if not segment_translations:
                continue
            
            # Refine each block with full segment context
            for block in translatable:
                if not block.translated_text:
                    continue
                
                try:
                    # Build refinement prompt
                    refined = self._refine_block_translation(
                        block,
                        segment_context=segment_translations
                    )
                    
                    if refined and refined != block.translated_text:
                        # Validate constraints before accepting
                        if self.config.validate_refinement_constraints:
                            if self._validate_refinement_constraints(block, refined):
                                block.translated_text = refined
                                refinement_stats['blocks_refined'] += 1
                                logger.debug(f"Refined block {block.block_id}")
                            else:
                                logger.warning(
                                    f"Refinement violated constraints for block {block.block_id}, "
                                    f"keeping original"
                                )
                                refinement_stats['constraint_violations'] += 1
                        else:
                            # No validation: accept refinement
                            block.translated_text = refined
                            refinement_stats['blocks_refined'] += 1
                    
                except Exception as e:
                    logger.error(f"Refinement failed for block {block.block_id}: {e}")
                    refinement_stats['blocks_failed'] += 1
                    # Keep original translation
        
        # Log statistics
        logger.info(
            f"Refinement complete: {refinement_stats['blocks_refined']} blocks refined, "
            f"{refinement_stats['constraint_violations']} violations avoided, "
            f"{refinement_stats['blocks_failed']} failures"
        )
        
        # Store in result metadata
        result.metadata.refinement_stats = refinement_stats
    
    def _refine_block_translation(
        self,
        block: Block,
        segment_context: List[str]
    ) -> Optional[str]:
        """Refine a single block's translation for coherence.
        
        Args:
            block: Block to refine
            segment_context: Other translations in same segment for context
        
        Returns:
            Refined translation or None if refinement fails
        """
        # Build refinement prompt
        context_str = "\n".join([f"- {t[:100]}" for t in segment_context[:5]])
        
        refinement_prompts = {
            "coherence": (
                "Improve the following translation for coherence and fluency while "
                "preserving all placeholders (<MATH_X>, <URL_X>, etc.) and technical terminology exactly. "
                "Do not modify any content in angle brackets."
            ),
            "style": (
                "Improve the following translation for academic style while "
                "preserving all placeholders and terminology exactly."
            ),
            "terminology": (
                "Ensure the following translation uses correct scientific terminology while "
                "preserving all placeholders exactly."
            )
        }
        
        prompt_type = self.config.refinement_prompt
        refinement_instruction = refinement_prompts.get(prompt_type, refinement_prompts['coherence'])
        
        system_prompt = f"""{refinement_instruction}

Context (other translations in this section):
{context_str}

Current translation to refine:
{block.translated_text}

Provide only the refined translation, with no explanations."""
        
        user_prompt = block.translated_text
        
        # Use refinement backend if specified, otherwise use main backend
        if self.config.refinement_backend:
            # Save current backend
            original_backend = self.config.backend
            original_translator = self.translator
            
            try:
                # Temporarily switch to refinement backend
                self.translator = self._create_translator(backend_override=self.config.refinement_backend)
                refined = self._call_translator(system_prompt, user_prompt, temperature=0.5)
                return refined
            finally:
                # Restore original
                self.translator = original_translator
        else:
            # Use main backend
            return self._call_translator(system_prompt, user_prompt, temperature=0.5)
    
    def _validate_refinement_constraints(self, block: Block, refined_text: str) -> bool:
        """Validate that refinement preserves placeholders and glossary terms.
        
        Args:
            block: Original block with constraints
            refined_text: Refined translation to validate
        
        Returns:
            True if constraints preserved, False otherwise
        """
        # Check 1: All placeholders must be preserved
        if block.masks:
            for mask in block.masks:
                if mask.placeholder not in refined_text:
                    logger.warning(
                        f"Refinement lost placeholder {mask.placeholder} "
                        f"(type: {mask.mask_type})"
                    )
                    return False
        
        # Check 2: Glossary terms must be preserved (if glossary enabled)
        if self.glossary_manager and block.source_text:
            # Find terms in source
            source_terms = self.glossary_manager.find_terms_in_text(block.source_text)
            
            if source_terms:
                # Check all expected translations are in refined text
                for term in source_terms:
                    if term.target.lower() not in refined_text.lower():
                        logger.warning(
                            f"Refinement lost glossary term: {term.target} "
                            f"(from {term.source})"
                        )
                        return False
        
        return True
    
    def _should_disable_masking(self) -> bool:
        """Check if masking should be disabled for this backend.
        
        Free translation services don't preserve mask placeholders,
        so masking can cause content loss. Better to skip masking
        and let the service translate everything naturally.
        """
        # These backends cannot reliably preserve mask tokens
        mask_unsafe_backends = {'cascade', 'free', 'mymemory', 'lingva'}
        return self.config.backend.lower() in mask_unsafe_backends
    
    def _translate_blocks_batch(self, document: Document):
        """Translate blocks using fast async batch processing."""
        from scitran.utils.fast_translator import FastTranslator
        
        # Create fast translator with async + caching
        # Optimize for speed: increase concurrency and reduce delays
        # Free services can handle more concurrent requests with proper caching
        max_concurrent = 5 if self.config.backend.lower() in {'cascade', 'free'} else 8
        
        fast_translator = FastTranslator(
            max_concurrent=max_concurrent,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else ".cache/translations",
            timeout=30,  # Increased timeout for reliability
            retry_count=3,  # More retries
            rate_limit_delay=0.2  # Reduced delay - caching handles most requests
        )
        
        blocks = document.translatable_blocks
        total_blocks = len(blocks)
        
        self._report_progress(0.2, f"Batch translating {total_blocks} blocks...")
        
        def progress_callback(completed: int, total: int):
            progress = 0.2 + (0.6 * completed / max(total, 1))
            self._report_progress(progress, f"Fast mode: {completed}/{total} blocks")
        
        # Get texts to translate
        texts = []
        for block in blocks:
            text = block.masked_text or block.source_text
            texts.append(text)
        
        # Fast batch translate (async + caching + deduplication)
        translations = fast_translator.translate_batch_sync(
            texts=texts,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            progress_callback=progress_callback
        )
        
        # Apply translations with fallback for any missing results
        failed_block_ids = []
        for block, translation in zip(blocks, translations):
            source_text = block.masked_text or block.source_text
            final_translation = self._postprocess_translation(translation) if translation else translation
            
            # Treat empty or identical translations as failures to avoid partial output
            missing_or_identical = (
                not final_translation
                or not str(final_translation).strip()
                or str(final_translation).strip() == source_text.strip()
            )
            
            if missing_or_identical:
                final_translation = self._fallback_translation(block, [])
                if not final_translation or str(final_translation).strip() == source_text.strip():
                    failed_block_ids.append(block.block_id)
                else:
                    block.translated_text = final_translation
                    self.stats['blocks_succeeded'] += 1
            else:
                block.translated_text = final_translation
                self.stats['blocks_succeeded'] += 1
            
            self.stats['blocks_processed'] += 1
        
        if failed_block_ids:
            # Count failures before retry so we can adjust stats when retries succeed
            self.stats['blocks_failed'] += len(failed_block_ids)
            self._report_progress(
                0.81,
                f"Retry needed for blocks (fallback failed): {len(failed_block_ids)}"
            )
            self._retry_failed_blocks(document, failed_block_ids)
        
        # Store fast translator stats
        fast_stats = fast_translator.get_stats()
        self.stats['batch_cache_hits'] = fast_stats.get('cached', 0)
        self.stats['batch_translated'] = fast_stats.get('translated', 0)
        self.stats['batch_deduplicated'] = fast_stats.get('deduplicated', 0)
        
        cached = fast_stats.get('cached', 0)
        translated = fast_stats.get('translated', 0)
        self._report_progress(0.8, f"Done: {cached} cached, {translated} new translations")
        self._dbg_log(
            "H3",
            "pipeline:batch_done",
            "batch translation finished",
            {
                "cached": cached,
                "translated": translated,
                "failed_blocks": self.stats.get("blocks_failed", 0),
            },
        )
    
    def _translate_blocks_sequential(self, document: Document):
        """Translate blocks sequentially with caching support."""
        total_blocks = len(document.translatable_blocks)
        
        for i, block in enumerate(document.translatable_blocks):
            progress = 0.2 + (0.6 * (i + 1) / max(total_blocks, 1))
            self._report_progress(progress, f"Block {i+1}/{total_blocks}")
            
            try:
                text_to_translate = block.masked_text or block.source_text
                
                # Check cache first
                cached_translation = None
                if self._translation_cache:
                    cached_translation = self._translation_cache.get(
                        text_to_translate,
                        self.config.source_lang,
                        self.config.target_lang
                    )
                    if cached_translation:
                        self.stats['cache_hits'] += 1
                        block.translated_text = cached_translation
                        self.stats['blocks_succeeded'] += 1
                        self.stats['blocks_processed'] += 1
                        continue  # Skip to next block
                    else:
                        self.stats['cache_misses'] += 1
                
                # Get context for this block
                context = self._get_block_context(document, block)
                
                # Generate translation candidates
                candidates = self._generate_candidates(block, context)
                
                # Rerank and select best
                if self.config.enable_reranking and len(candidates) > 1:
                    best_translation, scored = self.reranker.rerank(
                        candidates=candidates,
                        block=block,
                        glossary=self.glossary,
                        context=context,
                        require_all_masks=self.config.validate_mask_restoration
                    )
                    
                    # Check if reranking improved quality
                    if scored and scored[0].candidate_id != 0:
                        self.stats['reranking_improvements'] += 1
                else:
                    best_translation = candidates[0] if candidates else ""
                    
                # Validate quality
                if self._validate_block_quality(block, best_translation):
                    block.translated_text = best_translation
                    self.stats['blocks_succeeded'] += 1
                    
                    # Cache the successful translation
                    if self._translation_cache and best_translation:
                        self._translation_cache.set(
                            text_to_translate,
                            self.config.source_lang,
                            self.config.target_lang,
                            best_translation
                        )
                else:
                    self.stats['quality_failures'] += 1
                    # Retry with different strategy or mark as failed
                    block.translated_text = self._fallback_translation(block, context)
                    
                # Update metadata
                block.metadata = TranslationMetadata(
                    backend=self.config.backend,
                    timestamp=datetime.now(),
                    duration=0,  # Would be tracked per block
                    glossary_terms_used=set(self.glossary.keys()),
                    context_used=len(context) > 0,
                    candidates_generated=len(candidates),
                    reranking_applied=self.config.enable_reranking
                )
                
            except Exception as e:
                logger.error(f"Failed to translate block {block.block_id}: {e}")
                self.stats['blocks_failed'] += 1
                
            self.stats['blocks_processed'] += 1
            
    def _generate_candidates(self, block: Block, context: List[Tuple[str, str]]) -> List[str]:
        """Generate translation candidates for a block."""
        candidates = []
        
        # For simple backends (cascade, free), just use plain text
        simple_backends = {'cascade', 'free', 'huggingface'}
        use_simple_prompt = self.config.backend.lower() in simple_backends
        
        if use_simple_prompt:
            # Simple backends just translate the text directly
            text_to_translate = block.masked_text or block.source_text
            system_prompt = None
            user_prompt = text_to_translate
        elif self.prompt_optimizer:
            # Advanced backends get the full prompt with context
            system_prompt, user_prompt = self.prompt_optimizer.generate_prompt(
                block=block,
                template_name=self.config.prompt_template,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                glossary_terms=self.glossary,
                context=context
            )
        else:
            # Fallback to simple prompt
            system_prompt = f"Translate from {self.config.source_lang} to {self.config.target_lang}"
            user_prompt = block.masked_text or block.source_text
            
        # Generate candidates
        for i in range(self.config.num_candidates):
            candidate = self._call_translator(system_prompt, user_prompt, temperature=0.3 + i*0.2)
            candidates.append(candidate)
            
        self.stats['total_candidates'] += len(candidates)
        
        return candidates
    
    def _call_translator(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Call the translation backend (SPRINT 3: with glossary injection)."""
        if self.translator is None:
            logger.warning("No translator configured, returning original text")
            return user_prompt
        
        try:
            from scitran.translation.base import TranslationRequest
            
            # SPRINT 3: Inject glossary terms into prompt
            enhanced_system_prompt = system_prompt
            if self.glossary_manager and system_prompt:
                glossary_section = self.glossary_manager.generate_prompt_section(user_prompt)
                if glossary_section:
                    enhanced_system_prompt = f"{system_prompt}\n\n{glossary_section}"
                    logger.debug(f"Injected {len(self.glossary_manager.find_terms_in_text(user_prompt))} glossary terms into prompt")
            
            # Create translation request
            request = TranslationRequest(
                text=user_prompt,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                system_prompt=enhanced_system_prompt,
                temperature=temperature,
                glossary=self.glossary  # Keep legacy dict for compatibility
            )
            
            # Call translator
            response = self.translator.translate_sync(request)
            
            if response.translations:
                return self._postprocess_translation(response.translations[0])
            else:
                logger.warning("No translation returned from backend")
                return user_prompt
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return user_prompt
    
    def _get_block_context(self, document: Document, block: Block) -> List[Tuple[str, str]]:
        """Get translation context for a block."""
        if not self.config.enable_context:
            return []
            
        if self.config.use_extended_context:
            # Get all previous translations in document
            return document.get_previous_translations(
                block.block_id, 
                limit=self.config.context_window_size * 2
            )
        else:
            # Get previous translations in same segment
            segment = next((s for s in document.segments 
                          if block in s.blocks), None)
            if segment:
                window = segment.get_context_window(
                    block.block_id,
                    window_size=self.config.context_window_size
                )
                return [(b.source_text, b.translated_text) 
                       for b in window 
                       if b.translated_text and b.block_id != block.block_id]
            return []
    
    def _validate_block_quality(self, block: Block, translation: str) -> bool:
        """Validate if translation meets quality threshold."""
        if not translation:
            return False
        
        # Reject unchanged text when translating between different languages
        source_text = block.masked_text or block.source_text
        if (
            self.config.source_lang != self.config.target_lang
            and translation.strip() == source_text.strip()
        ):
            return False
        
        # Check mask preservation
        if block.masks:
            missing_masks = [m for m in block.masks if m.placeholder not in translation]
            if missing_masks:
                self.stats['mask_violations'] += 1
                if self.config.validate_mask_restoration:
                    return False
                    
        # Check minimum quality score (if reranking was used)
        if hasattr(block, 'metadata') and block.metadata:
            # Handle both dict and object metadata
            confidence = block.metadata.get('confidence', 1.0) if isinstance(block.metadata, dict) else block.metadata.confidence
            if confidence < self.config.quality_threshold:
                return False
                
        return True
    
    def _fallback_translation(self, block: Block, context: List[Tuple[str, str]]) -> str:
        """Generate fallback translation when primary fails - actually translate!"""
        text = block.masked_text or block.source_text
        
        # Try direct translation without fancy prompts
        try:
            from scitran.translation.base import TranslationRequest
            
            if self.translator:
                request = TranslationRequest(
                    text=text,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    temperature=0.3
                )
                response = self.translator.translate_sync(request)
                if response.translations and response.translations[0]:
                    self.stats['blocks_succeeded'] += 1
                    return self._postprocess_translation(response.translations[0])
        except Exception as e:
            logger.warning(f"Fallback translation failed: {e}")
        
        # Last resort: return original text (better than placeholder)
        return text

    # ---------- Helpers ----------
    def _dbg_log(self, hid: str, loc: str, message: str, data: Dict[str, Any]):
        """Write debug info to session log (used in debug mode)."""
        if not getattr(self.config, "debug_mode", False):
            return
        log_path = "/Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW/.cursor/debug.log"
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            payload = {
                "sessionId": "debug-session",
                "runId": "post-fix",
                "hypothesisId": hid,
                "location": loc,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _strip_control_chars(self, text: str) -> str:
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    def _apply_glossary(self, text: str) -> str:
        """Enforce glossary terms with simple longest-first replacement."""
        if not self.glossary:
            return text
        sorted_terms = sorted(self.glossary.items(), key=lambda kv: len(kv[0]), reverse=True)
        out = text
        for src, tgt in sorted_terms:
            if not src or not tgt:
                continue
            out = out.replace(src, tgt)
        return out

    def _postprocess_translation(self, translation: str) -> str:
        """Clean and enforce rules on translations."""
        if not translation:
            return translation
        cleaned = translation.strip()
        if cleaned.startswith("<think>"):
            end_idx = cleaned.find("</think>")
            if end_idx != -1:
                cleaned = cleaned[end_idx + len("</think>") :].strip()
        cleaned = self._strip_control_chars(cleaned)
        cleaned = self._apply_glossary(cleaned)
        return cleaned
    
    def _validate_translation(self, document: Document) -> Dict[str, float]:
        """Validate overall translation quality (SPRINT 3: Enhanced with glossary validation)."""
        metrics = {}
        
        # SPRINT 3: Calculate glossary adherence using GlossaryManager
        if self.glossary_manager:
            total_found = 0
            total_enforced = 0
            
            for block in document.translatable_blocks:
                if block.translated_text and block.source_text:
                    stats = self.glossary_manager.validate_translation(
                        block.source_text,
                        block.translated_text
                    )
                    total_found += stats.terms_found
                    total_enforced += stats.terms_enforced
            
            metrics['glossary_adherence'] = (total_enforced / total_found) if total_found > 0 else 1.0
            metrics['glossary_terms_found'] = total_found
            metrics['glossary_terms_enforced'] = total_enforced
            
        # Calculate layout preservation (simplified)
        blocks_with_layout = [b for b in document.all_blocks if b.bbox]
        if blocks_with_layout:
            preserved = sum(1 for b in blocks_with_layout if b.translated_text)
            metrics['layout_preservation'] = preserved / len(blocks_with_layout)
        else:
            metrics['layout_preservation'] = 1.0
            
        # BLEU and chrF would require reference translations
        metrics['bleu'] = None
        metrics['chrf'] = None
        
        return metrics
    
    def _setup_glossary_manager(self):
        """SPRINT 3: Setup glossary manager with configured domains.
        
        Returns:
            GlossaryManager instance or None
        """
        from scitran.translation.glossary.manager import GlossaryManager
        
        # Use provided manager if available
        if self.config.glossary_manager:
            logger.info("Using pre-configured glossary manager")
            return self.config.glossary_manager
        
        # Create new manager
        manager = GlossaryManager()
        
        # Load configured domains
        direction = f"{self.config.source_lang}-{self.config.target_lang}"
        for domain in self.config.glossary_domains:
            try:
                count = manager.load_domain(domain, direction)
                logger.info(f"Loaded {count} terms from {domain} glossary")
            except Exception as e:
                logger.warning(f"Could not load {domain} glossary: {e}")
        
        # Load custom glossary file if specified
        if self.config.glossary_path:
            try:
                count = manager.load_from_file(self.config.glossary_path)
                logger.info(f"Loaded {count} custom terms from {self.config.glossary_path}")
            except Exception as e:
                logger.warning(f"Could not load custom glossary: {e}")
        
        return manager if len(manager) > 0 else None
    
    def _load_glossary(self, document: Document) -> Dict[str, str]:
        """Legacy method: Load glossary as dictionary (deprecated, use manager instead)."""
        if self.glossary_manager:
            return self.glossary_manager.to_dict()
        
        # Fallback: empty glossary
        return {}
    
    def _create_translator(self, backend_override: Optional[str] = None):
        """Create translator instance based on backend.
        
        Args:
            backend_override: Use this backend instead of config.backend (for fallback)
        """
        backend_name = (backend_override or self.config.backend).lower()
        
        try:
            if backend_name == "cascade":
                from scitran.translation.backends.cascade_backend import CascadeBackend
                return CascadeBackend()
            elif backend_name == "free":
                from scitran.translation.backends.free_backend import FreeBackend
                return FreeBackend()
            elif backend_name == "huggingface":
                from scitran.translation.backends.huggingface_backend import HuggingFaceBackend
                return HuggingFaceBackend(model=self.config.model_name)
            elif backend_name == "ollama":
                from scitran.translation.backends.ollama_backend import OllamaBackend
                return OllamaBackend(model=self.config.model_name)
            elif backend_name == "openai":
                from scitran.translation.backends.openai_backend import OpenAIBackend
                return OpenAIBackend(api_key=self.config.api_key, model=self.config.model_name)
            elif backend_name == "anthropic":
                from scitran.translation.backends.anthropic_backend import AnthropicBackend
                return AnthropicBackend(api_key=self.config.api_key, model=self.config.model_name)
            elif backend_name == "deepseek":
                from scitran.translation.backends.deepseek_backend import DeepSeekBackend
                return DeepSeekBackend(api_key=self.config.api_key, model=self.config.model_name)
            else:
                logger.warning(f"Unknown backend: {backend_name}, using cascade")
                from scitran.translation.backends.cascade_backend import CascadeBackend
                return CascadeBackend()
        except ImportError as e:
            logger.error(f"Could not import backend {backend_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating backend {backend_name}: {e}")
            return None
    
    def _build_translation_request(self, text: str):
        """Build translation request for backend.
        
        Args:
            text: Text to translate
        
        Returns:
            TranslationRequest object
        """
        from scitran.translation.base import TranslationRequest
        
        return TranslationRequest(
            text=text,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            num_candidates=1,  # For coverage guarantee, we only need one
            context=None,
            glossary=self.glossary if self.config.enable_glossary else None
        )
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if provided."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        logger.info(f"[{progress:.0%}] {message}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        stats = dict(self.stats)
        
        # Add cache statistics from translation cache
        if self._translation_cache:
            cache_stats = self._translation_cache.stats()
            stats['cache_size'] = cache_stats.get('size', 0)
            stats['cache_type'] = cache_stats.get('type', 'unknown')
        
        # Combine batch cache hits with regular cache hits for reporting
        total_cache_hits = stats.get('cache_hits', 0) + stats.get('batch_cache_hits', 0)
        if total_cache_hits > 0:
            stats['total_cache_hits'] = total_cache_hits
        
        # Add component statistics
        if self.masking_engine:
            stats['masking'] = self.masking_engine.get_statistics()
            
        if self.reranker:
            stats['reranking'] = self.reranker.get_reranking_stats()
            
        if self.prompt_optimizer:
            stats['prompt_optimization'] = {
                'rounds': self.prompt_optimizer.optimization_rounds,
                'best_template': self.prompt_optimizer.select_best_template()
            }
            
        return stats

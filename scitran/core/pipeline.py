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
import re
import os
import unicodedata

from scitran.core.models import (
    Document, Block, Segment, TranslationResult,
    TranslationMetadata, BlockType
)
from scitran.core.validator import TranslationCompletenessValidator, ValidationResult
from scitran.core.artifacts import ArtifactGenerator
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
    # DEFAULT: deepseek (best quality/price ratio)
    backend: str = "deepseek"  # deepseek, openai, anthropic, ollama, free
    model_name: Optional[str] = None  # Specific model to use
    api_key: Optional[str] = None
    
    # Innovation #1: Masking configuration
    enable_masking: bool = True
    masking_config: MaskingConfig = field(default_factory=MaskingConfig)
    validate_mask_restoration: bool = True
    mask_custom_macros: bool = True
    mask_apostrophes_in_latex: bool = True
    
    # Innovation #2: Document context
    enable_context: bool = True
    context_window_size: int = 5
    use_extended_context: bool = False  # Use context from entire document
    
    # Innovation #3: Layout preservation
    preserve_layout: bool = True
    layout_detection_method: str = "yolo"  # yolo, heuristic, hybrid
    
    # STEP 2: Table/Figure text translation policy
    translate_table_text: bool = True  # Translate text inside tables
    translate_figure_text: bool = True  # Translate text inside figures (captions, labels)
    
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
    allow_partial: bool = False  # Allow partial output (NOT RECOMMENDED - breaks quality)
    max_translation_retries: int = 3  # Retry failed blocks this many times
    retry_backoff_factor: float = 2.0  # Exponential backoff multiplier
    enable_fallback_backend: bool = True  # Escalate to stronger backend on failure
    fallback_backend: str = "openai"  # Backend to use for failed blocks (openai is strongest fallback)
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
    batch_size: int = 20  # Blocks to translate in parallel (increased for speed)
    cache_translations: bool = True
    cache_dir: Optional[Path] = None
    max_retries: int = 3
    timeout: int = 30  # Seconds per block
    enable_parallel_processing: bool = True  # Enable parallel processing for large documents
    max_workers: Optional[int] = None  # Max parallel workers (None = auto-detect)
    adaptive_concurrency: bool = True  # Adjust concurrency based on backend
    
    # Fast mode (PHASE 1.3): Optimized for speed over quality
    # DISABLED BY DEFAULT: Quality over speed is the priority
    fast_mode: bool = False  # When True: num_candidates=1, no reranking, no context, higher concurrency
    
    # Output settings
    output_format: str = "pdf"  # pdf, docx, txt, json
    debug_mode: bool = False
    log_level: str = "INFO"
    debug_log_path: Optional[Path] = None  # PHASE 4.1: Path for structured debug logs (JSONL)
    generate_artifacts: bool = True  # Generate extraction/masking/translation JSON artifacts
    artifact_dir: Optional[Path] = None  # Directory for artifacts (default: artifacts/<run_id>)
    
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
            
        # API key validation moved to _create_translator (checks env vars too)
        # No longer validate here to avoid false positives
            
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
        
        # Apply fast_mode defaults if enabled
        if self.config.fast_mode:
            logger.info("Fast mode enabled: optimizing for speed")
            self.config.num_candidates = 1
            self.config.enable_reranking = False
            self.config.enable_context = False
            self.config.context_window_size = 0
            # Increase concurrency for speed
            if not self.config.max_workers:
                self.config.max_workers = 10  # Higher default for fast mode
        
        # For free/cascade backends, force efficient settings (they're deterministic)
        simple_backends = {'cascade', 'free', 'huggingface', 'libre', 'argos', 'mymemory'}
        if self.config.backend.lower() in simple_backends:
            if self.config.num_candidates > 1:
                logger.info(f"Backend '{self.config.backend}' is deterministic - forcing num_candidates=1")
                self.config.num_candidates = 1
            if self.config.enable_reranking:
                logger.info(f"Backend '{self.config.backend}' doesn't benefit from reranking - disabling")
                self.config.enable_reranking = False
        
        # Sync masking flags into masking_config
        if self.config.masking_config:
            self.config.masking_config.mask_custom_macros = self.config.mask_custom_macros
            self.config.masking_config.mask_apostrophes_in_latex = self.config.mask_apostrophes_in_latex
        else:
            self.config.masking_config = MaskingConfig(
                mask_custom_macros=self.config.mask_custom_macros,
                mask_apostrophes_in_latex=self.config.mask_apostrophes_in_latex,
            )
        
        # PHASE 4.1: Setup debug logging
        self._debug_log_file = None
        if self.config.debug_mode and self.config.debug_log_path:
            debug_path = Path(self.config.debug_log_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_log_file = open(debug_path, 'a', encoding='utf-8')
            logger.info(f"Debug logging enabled: {debug_path}")
        elif self.config.debug_mode:
            # Default path if debug_mode but no path specified
            default_path = Path(".cache/scitrans/debug.jsonl")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_log_file = open(default_path, 'a', encoding='utf-8')
            logger.info(f"Debug logging enabled: {default_path}")
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            raise ValueError(f"Configuration issues: {', '.join(issues)}")
            
        # Initialize components
        self.masking_engine = MaskingEngine(self.config.masking_config) if self.config.enable_masking else None
        self.prompt_optimizer = PromptOptimizer() if self.config.optimize_prompts else None
        self.reranker = AdvancedReranker(strategy=self.config.reranking_strategy) if self.config.enable_reranking else None
        self.validator = TranslationCompletenessValidator(
            require_full_coverage=not self.config.allow_partial,
            require_mask_preservation=self.config.validate_mask_restoration,
            require_no_identity=self.config.detect_identity_translation,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang
        )
        
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
        
        # Progress callback is already set in __init__, don't overwrite
        
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
        
        # Initialize result
        result = TranslationResult(
            document=document,
            backend_used=self.config.backend
        )
        
        # Initialize artifact generator if enabled
        artifacts = None
        if self.config.generate_artifacts:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_dir = self.config.artifact_dir or Path(f"artifacts/{run_id}")
            artifacts = ArtifactGenerator(run_id, artifact_dir)
            artifacts.log(f"Starting translation: {document.document_id}")
        
        try:
            # Setup
            self._setup_translation(document)
            
            # Save extraction artifacts
            if artifacts:
                artifacts.save_extraction(document)
            
            # STRICT MASKING: Never disable masking (requirement)
            # Free services MUST preserve mask tokens or translation fails
            use_masking = self.config.enable_masking and not self.config.ablation_disable_masking
            
            # Phase 1: Apply masking (Innovation #1)
            if use_masking:
                self._report_progress(0.1, "Applying content masking...")
                self.masking_engine.reset()
                document = self.masking_engine.mask_document(document)
                result.masks_applied = document.stats.get('total_masks', 0)
                
                # Save masking artifacts
                if artifacts:
                    artifacts.save_masking(document, self.masking_engine)
            
            # Phase 2: Translate blocks with context (Innovation #2)
            self._report_progress(0.2, "Translating document...")
            self._translate_all_blocks(document)
            
            # Phase 2.5: Pre-unmask validation (coverage check only)
            self._report_progress(0.65, "Checking translation coverage...")
            # Quick check: do all translatable blocks have non-empty translations?
            pre_validation_errors = []
            for block in document.translatable_blocks:
                if not block.translated_text or not block.translated_text.strip():
                    pre_validation_errors.append(f"Block {block.block_id}: Missing translation")
            
            if pre_validation_errors and not self.config.allow_partial:
                logger.warning(f"Pre-validation found {len(pre_validation_errors)} missing translations")
                # Will be handled by repair loop after unmask
            
            # Phase 2.75: SPRINT 4 - Document-level refinement (if enabled)
            if self.config.enable_refinement and not self.config.ablation_disable_refinement:
                self._report_progress(0.7, "Refining translation coherence...")
                self._refine_document_translation(document, result)
            
            # Phase 3: Restore masks with TOLERANT validation
            if use_masking:
                self._report_progress(0.75, "Restoring masked content (tolerant)...")
                document = self.masking_engine.unmask_document(
                    document, 
                    validate=True  # Use tolerant unmasking
                )
                result.masks_restored = result.masks_applied - len(self.masking_engine.validation_errors)
            
            # Save translation artifacts (before final validation)
            if artifacts:
                artifacts.log("Translation phase complete, starting validation...")
            
            # Phase 4: FINAL validation POST-unmask (STRICT)
            self._report_progress(0.85, "Validating translation completeness...")
            validation_result = self.validator.validate(document)
            
            # If validation fails, attempt repair
            if not validation_result.is_valid:
                logger.warning(f"Final validation failed: {len(validation_result.errors)} errors")
                self._report_progress(0.87, f"Repairing {len(validation_result.failed_blocks)} failed blocks...")
                
                # Repair loop
                validation_result = self._repair_failed_blocks(document, validation_result, use_masking)
                
                # Final validation after repair
                if not validation_result.is_valid and not self.config.allow_partial:
                    # Hard failure - refuse to produce partial output
                    result.success = False
                    result.error = f"Translation incomplete: {validation_result.coverage:.1%} coverage"
                    result.validation_result = validation_result
                    
                    # Print validation report
                    logger.error(self.validator.generate_report(validation_result))
                    
                    if artifacts:
                        artifacts.log(f"VALIDATION FAILED: {len(validation_result.errors)} errors")
                        artifacts.save_translation(document, validation_result)
                    
                    raise ValueError(
                        f"Translation validation failed: {len(validation_result.errors)} errors. "
                        f"Coverage: {validation_result.coverage:.1%}. "
                        f"Set allow_partial=True to force output (NOT RECOMMENDED)."
                    )
            
            # Store validation result
            result.validation_result = validation_result
            result.coverage = validation_result.coverage
            
            # Final guard: ensure all translatable blocks have translated_text
            missing_blocks = [
                b for b in document.translatable_blocks
                if not b.translated_text or not b.translated_text.strip()
            ]
            if missing_blocks:
                logger.info(f"Final guard: {len(missing_blocks)} blocks missing translation, retrying best-effort.")
                self._ensure_all_translated(document, missing_blocks, use_masking)
                # Recompute coverage
                validation_result = self.validator.validate(document)
                result.validation_result = validation_result
                result.coverage = validation_result.coverage
                if not validation_result.is_valid and not self.config.allow_partial:
                    raise ValueError(
                        f"Translation incomplete after final guard: {len(validation_result.errors)} errors, "
                        f"coverage {validation_result.coverage:.1%}"
                    )
            
            # Save translation artifacts
            if artifacts:
                artifacts.save_translation(document, validation_result)
                
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
            
            # STRICT SUCCESS CRITERIA: Must pass validation
            if hasattr(result, 'validation_result') and result.validation_result:
                result.success = result.validation_result.is_valid
            else:
                # Fallback if validation not run
                result.success = result.blocks_failed == 0 and result.coverage >= 1.0
            
            self._report_progress(1.0, "Translation complete!")
            
        except Exception as e:
            import traceback
            logger.error(f"Pipeline error: {e}")
            if artifacts:
                artifacts.log(f"ERROR: {e}")
                artifacts.log(traceback.format_exc())
            result.success = False
            result.error = str(e)
        
        finally:
            # Close artifacts
            if artifacts:
                artifacts.close()
            
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
        else:
            self.glossary_manager = None
            self.glossary = {}
            
        # Initialize translator - fail fast if API key is missing
        try:
            self.translator = self._create_translator()
            if self.translator is None:
                raise ValueError(f"Failed to create translator for backend: {self.config.backend}")
        except ValueError as e:
            # API key errors should fail immediately
            raise ValueError(f"Backend configuration error: {e}")

        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}
    
    def _get_blocks_to_translate(self, document: Document) -> List[Block]:
        """
        Get blocks that should be translated based on policy flags.
        
        STEP 2: Respects translate_table_text and translate_figure_text flags.
        """
        blocks = []
        for block in document.translatable_blocks:
            # Check table policy
            if block.block_type == BlockType.TABLE and not self.config.translate_table_text:
                continue
            
            # Check figure policy
            if block.block_type == BlockType.FIGURE and not self.config.translate_figure_text:
                continue
            
            blocks.append(block)
        
        return blocks
        
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
        
        QUALITY OVER SPEED: Batch mode is disabled by default.
        Sequential mode provides:
        - Multi-candidate generation
        - Reranking for best quality
        - Context-aware translation
        - Better error handling
        
        Batch mode is only used for free backends that don't support quality features.
        """
        # Disable batch mode by default - quality over speed
        if self.config.fast_mode:
            # Fast mode enabled: use batch for speed
            return True
        
        # Only use batch for simple free backends that don't benefit from reranking
        simple_backends = {'cascade', 'free', 'libre', 'argos'}
        if self.config.backend.lower() in simple_backends:
            return True
        
        # For quality backends (deepseek, openai, anthropic), use sequential for better quality
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
    
    def _repair_failed_blocks(self, document: Document, validation_result: ValidationResult, use_masking: bool = True) -> ValidationResult:
        """
        Repair loop: retry only failed blocks with enhanced strategies.
        
        Strategies:
        1. Retry with smaller batch size (1 block at a time)
        2. Generate more candidates and rerank
        3. Use stricter prompts
        4. Apply mask-aware validation
        
        Args:
            document: Document with failed blocks
            validation_result: Initial validation result
        
        Returns:
            Updated validation result after repair
        """
        # Get failed blocks
        failed_block_ids = set(validation_result.failed_blocks + validation_result.identity_blocks)
        if not failed_block_ids:
            return validation_result
        
        logger.info(f"Starting repair loop for {len(failed_block_ids)} failed blocks")
        
        failed_blocks = [
            block for block in document.translatable_blocks
            if block.block_id in failed_block_ids
        ]
        
        # Retry strategy: More candidates + reranking for failed blocks
        original_num_candidates = self.config.num_candidates
        original_enable_reranking = self.config.enable_reranking
        
        # Override for repair: always use multiple candidates + reranking
        self.config.num_candidates = max(3, self.config.num_candidates)
        self.config.enable_reranking = True
        
        repaired = 0
        still_failed = []
        fallback_used = []
        
        for block in failed_blocks:
            repaired_this_block = False
            
            # Strategy A: Retry with primary backend (more candidates)
            try:
                context = self._get_block_context(document, block)
                candidates = self._generate_candidates(block, context)
                
                if candidates and any(c and c.strip() for c in candidates):
                    # Rerank if multiple candidates
                    if self.reranker and len(candidates) > 1:
                        best_translation, scored = self.reranker.rerank(
                            candidates=candidates,
                            block=block,
                            glossary=self.glossary,
                            context=context,
                            require_all_masks=False  # Don't check pre-unmask
                        )
                    else:
                        best_translation = candidates[0]
                    
                    # Basic checks
                    if best_translation and best_translation.strip():
                        # Check not identity (using smarter heuristic)
                        if not self._is_identity_failure(block.source_text, best_translation):
                            block.translated_text = best_translation
                            
                            # Re-unmask if using masking
                            if use_masking and block.masks:
                                self.masking_engine.unmask_block(block, validate=False)
                            
                            # Check restoration succeeded
                            if block.masks:
                                missing = getattr(block.metadata, 'missing_placeholders', []) if block.metadata else []
                                if not missing:
                                    repaired_this_block = True
                            else:
                                repaired_this_block = True
            except Exception as e:
                logger.debug(f"Primary retry failed for {block.block_id}: {e}")
            
            # Strategy B: For mask-related failures with free/cascade, try alnum placeholders
            if not repaired_this_block and block.masks and self.config.backend.lower() in {'free', 'cascade'}:
                try:
                    logger.info(f"Trying alnum placeholders for {block.block_id}")
                    # Re-mask with alnum style
                    original_style = self.config.masking_config.placeholder_style
                    self.config.masking_config.placeholder_style = "alnum"
                    
                    # Re-mask this block
                    self.masking_engine.reset()
                    temp_block = self.masking_engine.mask_block(block)
                    
                    # Translate
                    text = temp_block.masked_text or temp_block.source_text
                    from scitran.translation.base import TranslationRequest
                    request = TranslationRequest(
                        text=text,
                        source_lang=self.config.source_lang,
                        target_lang=self.config.target_lang,
                        num_candidates=1
                    )
                    response = self.translator.translate_sync(request)
                    
                    if response.translations and response.translations[0]:
                        temp_block.translated_text = response.translations[0]
                        # Unmask
                        self.masking_engine.unmask_block(temp_block, validate=False)
                        
                        # Check if restoration succeeded
                        missing = getattr(temp_block.metadata, 'missing_placeholders', []) if temp_block.metadata else []
                        if not missing:
                            block.translated_text = temp_block.translated_text
                            block.masks = temp_block.masks
                            repaired_this_block = True
                            logger.info(f"✓ Repaired {block.block_id} with alnum placeholders")
                    
                    # Restore original style
                    self.config.masking_config.placeholder_style = original_style
                except Exception as e:
                    logger.debug(f"Alnum retry failed for {block.block_id}: {e}")
            
            # Strategy C: Fallback to stronger backend
            if not repaired_this_block and self.config.enable_fallback_backend:
                try:
                    logger.info(f"Escalating {block.block_id} to fallback backend: {self.config.fallback_backend}")
                    
                    # Create or reuse fallback translator
                    if not hasattr(self, '_fallback_translator'):
                        self._fallback_translator = self._create_translator(backend_override=self.config.fallback_backend)
                    
                    text = block.masked_text or block.source_text
                    from scitran.translation.base import TranslationRequest
                    request = TranslationRequest(
                        text=text,
                        source_lang=self.config.source_lang,
                        target_lang=self.config.target_lang,
                        num_candidates=1,
                        system_prompt="You are a professional translator. IMPORTANT: Preserve all placeholder tokens (<<TYPE_XXXX>> or SCITRANS_TYPE_XXXX_SCITRANS) EXACTLY as they appear."
                    )
                    response = self._fallback_translator.translate_sync(request)
                    
                    if response.translations and response.translations[0]:
                        block.translated_text = response.translations[0]
                        
                        # Unmask if using masking
                        if use_masking and block.masks:
                            self.masking_engine.unmask_block(block, validate=False)
                        
                        # Check restoration
                        missing = getattr(block.metadata, 'missing_placeholders', []) if block.metadata else []
                        if not missing and not self._is_identity_failure(block.source_text, block.translated_text):
                            repaired_this_block = True
                            fallback_used.append(block.block_id)
                            if block.metadata:
                                block.metadata.backend_used = self.config.fallback_backend
                            logger.info(f"✓ Repaired {block.block_id} via fallback backend")
                except Exception as e:
                    logger.debug(f"Fallback failed for {block.block_id}: {e}")
            
            # Final accounting
            if repaired_this_block:
                repaired += 1
                if block.metadata:
                    block.metadata.status = "translated_ok"
                    block.metadata.retry_count = 1
            else:
                still_failed.append(block.block_id)
                if block.metadata:
                    block.metadata.status = "failed"
                    block.metadata.failure_reason = "All repair strategies exhausted"
        
        # Restore original config
        self.config.num_candidates = original_num_candidates
        self.config.enable_reranking = original_enable_reranking
        
        logger.info(f"Repair complete: {repaired} blocks repaired ({len(fallback_used)} via fallback), {len(still_failed)} still failed")
        
        # Re-validate after repair
        return self.validator.validate(document)

    def _ensure_all_translated(self, document: Document, missing_blocks: List[Block], use_masking: bool = True) -> None:
        """Best-effort translation for any remaining missing blocks."""
        from scitran.translation.base import TranslationRequest
        for block in missing_blocks:
            try:
                text = block.masked_text or block.source_text
                if not text or not text.strip():
                    continue
                request = TranslationRequest(
                    text=text,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    num_candidates=1,
                    system_prompt="You are a professional translator. Translate accurately and preserve placeholder tokens exactly."
                )
                resp = self.translator.translate_sync(request)
                if resp.translations and resp.translations[0]:
                    block.translated_text = resp.translations[0]
                    if use_masking and block.masks:
                        self.masking_engine.unmask_block(block, validate=False)
            except Exception as e:
                logger.debug(f"Final guard translation failed for {block.block_id}: {e}")
    
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
                # Get list of missing block IDs
                missing_block_ids = [block.block_id for block in missing_blocks]
                raise TranslationCoverageError(
                    coverage=coverage,
                    missing_blocks=missing_block_ids,
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
                    # Check if text has meaningful alphabetic content (not just numbers/symbols)
                    # Also require minimum length to avoid flagging short symbols/formulas
                    if self._has_alphabetic_content(source) and len(source.strip()) >= 5:
                        # Additional check: if it's mostly symbols/numbers, don't flag as identity
                        # (formulas, citations like [1], [2] might legitimately be identical)
                        alpha_ratio = sum(1 for c in source if c.isalpha()) / max(len(source), 1)
                        if alpha_ratio >= 0.4:  # At least 40% alphabetic characters
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
    
    def _is_suspected_partial(self, source: str, translated: str, finish_reason: Optional[str] = None) -> bool:
        """Detect if translation is likely truncated or partial (RELIABLE + CHEAP).
        
        Args:
            source: Source text
            translated: Translated text
            finish_reason: Backend finish reason (e.g., "length", "stop")
        
        Returns:
            True if translation appears truncated/partial
        """
        if not source or not translated:
            return False
        
        # Check 1: finish_reason indicates truncation (STRONG evidence)
        if finish_reason and finish_reason in ["length", "max_tokens", "truncated"]:
            logger.warning(f"Truncation detected: finish_reason={finish_reason}")
            return True
        
        # Check 2: Translated much shorter than source (suspicious) - BUT disable for CJK
        # CJK languages can be much shorter than English legitimately
        cjk_langs = {'zh', 'ja', 'ko', 'th'}
        is_cjk = any(lang in cjk_langs for lang in [self.config.source_lang, self.config.target_lang])
        
        if not is_cjk and len(source) > 300:
            # Very suspicious ratio for non-CJK
            if len(translated) < 0.15 * len(source):
                logger.warning(f"Suspicious length: source={len(source)}, translated={len(translated)}")
                return True
        
        # Check 3: Ends mid-sentence (strong evidence if also short ratio)
        if len(source) > 200 and len(translated) < 0.4 * len(source):
            # Check if ends without proper punctuation
            if translated.rstrip() and translated.rstrip()[-1] not in '.!?。！？':
                logger.warning(f"Ends mid-sentence + short ratio")
                return True
        
        # DON'T check for verbatim sentences - too many false positives
        
        return False
    
    def _chunk_for_translation(self, text: str) -> List[str]:
        """Chunk text for translation while preserving placeholders (PHASE 3.1).
        
        Args:
            text: Text to chunk
        
        Returns:
            List of chunks that can be translated separately
        """
        # Find all placeholders (<<MATH_XXXX>>, etc.)
        placeholder_pattern = r'<<[A-Z_]+_\d+>>'
        placeholders = re.findall(placeholder_pattern, text)
        
        # Split by paragraph boundaries first
        chunks = text.split('\n\n')
        if len(chunks) == 1:
            # No paragraph breaks, try sentence boundaries
            chunks = re.split(r'(?<=[.!?])\s+', text)
        
        # Ensure no chunk splits a placeholder
        safe_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            # Check if adding this chunk would split a placeholder
            test_text = current_chunk + " " + chunk if current_chunk else chunk
            
            # Count placeholders in test_text
            test_placeholders = re.findall(placeholder_pattern, test_text)
            
            # If all placeholders are complete, it's safe
            if all(ph in test_placeholders for ph in re.findall(placeholder_pattern, current_chunk + chunk)):
                current_chunk = test_text
            else:
                # Would split placeholder, save current and start new
                if current_chunk:
                    safe_chunks.append(current_chunk.strip())
                current_chunk = chunk
        
        if current_chunk:
            safe_chunks.append(current_chunk.strip())
        
        return [c for c in safe_chunks if c]
    
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
    
    def _is_identity_failure(self, source: str, translated: str, min_length: int = 30, min_words: int = 6) -> bool:
        """Determine if identical text represents a translation failure (smarter heuristic).
        
        Args:
            source: Source text
            translated: Translated text
            min_length: Minimum character length to check
            min_words: Minimum word count to check
        
        Returns:
            True if this is a REAL identity failure (should retry)
        """
        if not source or not translated:
            return False
        
        # Normalize for comparison
        source_norm = self._normalize_text(source)
        target_norm = self._normalize_text(translated)
        
        if source_norm != target_norm:
            return False  # Not identical, not a failure
        
        # Same text - determine if it's a problem
        
        # Check 1: Different languages? If en->fr and text is identical, likely a failure
        if self.config.source_lang == self.config.target_lang:
            return False  # Same language, identity is expected
        
        # Check 2: Too short or too few words? Likely proper nouns, titles, etc.
        if len(source.strip()) < min_length:
            return False  # Short text, identity OK
        
        words = source.split()
        if len(words) < min_words:
            return False  # Few words, likely title/heading, identity OK
        
        # Check 3: Mostly uppercase, numbers, symbols? Likely technical, identity OK
        alpha_chars = sum(1 for c in source if c.isalpha())
        upper_chars = sum(1 for c in source if c.isupper())
        digit_chars = sum(1 for c in source if c.isdigit())
        
        if alpha_chars > 0 and upper_chars / alpha_chars > 0.7:
            return False  # Mostly uppercase, likely acronyms/headers
        
        if len(source) > 0 and digit_chars / len(source) > 0.3:
            return False  # Lots of numbers, likely technical
        
        # Check 4: Mostly placeholders? Identity OK
        placeholder_pattern = r'<<[A-Z_]+_\d+>>|SCITRANS_[A-Z_]+_\d+_SCITRANS'
        placeholders = re.findall(placeholder_pattern, source)
        if placeholders:
            placeholder_chars = sum(len(p) for p in placeholders)
            if placeholder_chars / len(source) > 0.5:
                return False  # Mostly placeholders
        
        # If we get here: long text, enough words, not technical → TRUE failure
        return True
    
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
    
    def _translate_blocks_batch(self, document: Document):
        """
        Translate blocks using async batch processing with configured backend.
        
        FIXED: Now uses the actual configured translator backend instead of free services.
        """
        import asyncio
        from scitran.translation.base import TranslationRequest
        
        blocks = document.translatable_blocks
        total_blocks = len(blocks)
        
        if not self.translator:
            raise ValueError("No translator configured - cannot proceed with batch translation")
        
        self._report_progress(0.2, f"Batch translating {total_blocks} blocks...")
        
        # Calculate optimal concurrency
        backend = self.config.backend.lower()
        if self.config.adaptive_concurrency:
            if backend in {'cascade', 'free'}:
                base_concurrent = 5  # Lower for free backends (rate limits)
            elif backend in {'openai', 'anthropic', 'deepseek'}:
                base_concurrent = 10  # Higher for paid APIs
            else:
                base_concurrent = 6
            max_concurrent = min(base_concurrent, self.config.max_workers or base_concurrent)
        else:
            max_concurrent = self.config.max_workers or 5
        
        # Build translation requests
        requests = []
        for block in blocks:
            text = block.masked_text or block.source_text
            if not text or not text.strip():
                continue
            
            # Build proper translation request with system prompt
            system_prompt = None
            if self.prompt_optimizer:
                context = self._get_block_context(document, block)
                system_prompt, _ = self.prompt_optimizer.generate_prompt(
                    block=block,
                    template_name=self.config.prompt_template,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    glossary_terms=self.glossary,
                    context=context
                )
            
            request = TranslationRequest(
                text=text,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                system_prompt=system_prompt,
                temperature=self.config.temperature if hasattr(self.config, 'temperature') else 0.0,
                num_candidates=1,
                glossary=self.glossary if self.config.enable_glossary else None
            )
            requests.append((block, request))
        
        # Async batch translation with progress reporting
        def progress_callback(completed: int, total: int):
            progress = 0.2 + (0.6 * completed / max(total, 1))
            self._report_progress(progress, f"Translating: {completed}/{total} blocks")
        
        async def translate_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            results = {}
            
            async def translate_one(block, req):
                async with semaphore:
                    try:
                        # Most backends use sync - run in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        if hasattr(self.translator, 'translate'):
                            # Backend has async support
                            response = await self.translator.translate(req)
                        else:
                            # Backend only has sync - run in executor
                            response = await loop.run_in_executor(
                                None, 
                                self.translator.translate_sync, 
                                req
                            )
                        
                        if response.translations and response.translations[0]:
                            translation = response.translations[0]
                            # Clean output
                            translation = self._postprocess_translation(translation)
                            results[block.block_id] = translation
                            
                            # Report progress
                            completed = sum(1 for r in results.values() if r is not None)
                            progress_callback(completed, len(requests))
                        else:
                            results[block.block_id] = None
                    except Exception as e:
                        logger.error(f"Batch translation failed for {block.block_id}: {e}")
                        results[block.block_id] = None
            
            # Run all translations concurrently
            await asyncio.gather(*[translate_one(block, req) for block, req in requests])
            return results
        
        # Run async batch
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        translations_dict = loop.run_until_complete(translate_batch())
        
        # Apply translations to blocks
        translations = []
        for block in blocks:
            translation = translations_dict.get(block.block_id)
            translations.append(translation)
        
        # Apply translations with fallback for any missing results
        failed_block_ids = []

        blocks_by_page_before = {}
        blocks_by_page_after = {}

        for block, translation in zip(blocks, translations):

            page_num = block.bbox.page if block.bbox else -1
            if page_num not in blocks_by_page_before:
                blocks_by_page_before[page_num] = {"total": 0, "has_translation": 0}
            blocks_by_page_before[page_num]["total"] += 1

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

            if page_num not in blocks_by_page_after:
                blocks_by_page_after[page_num] = {"total": 0, "translated": 0, "failed": 0}
            blocks_by_page_after[page_num]["total"] += 1
            if block.translated_text:
                blocks_by_page_after[page_num]["translated"] += 1
            else:
                blocks_by_page_after[page_num]["failed"] += 1

            self.stats['blocks_processed'] += 1

        if failed_block_ids:
            # Count failures before retry so we can adjust stats when retries succeed
            self.stats['blocks_failed'] += len(failed_block_ids)
            self._report_progress(
                0.81,
                f"Retry needed for blocks (fallback failed): {len(failed_block_ids)}"
            )
            self._retry_failed_blocks(document, failed_block_ids)
        
        # Report completion
        succeeded = sum(1 for b in blocks if b.translated_text)
        self._report_progress(0.8, f"Done: {succeeded}/{total_blocks} blocks translated")
    
    def _translate_blocks_sequential(self, document: Document):
        """Translate blocks sequentially with caching support."""
        total_blocks = len(document.translatable_blocks)
        
        # Get page info for enhanced progress
        pages = set()
        for block in document.translatable_blocks:
            if block.bbox and block.bbox.page is not None:
                pages.add(block.bbox.page)
        total_pages = len(pages) if pages else 1
        
        for i, block in enumerate(document.translatable_blocks):
            progress = 0.2 + (0.6 * (i + 1) / max(total_blocks, 1))
            page_num = block.bbox.page if block.bbox and block.bbox.page is not None else None
            self._report_progress(
                progress, 
                f"Translating block {i+1}/{total_blocks}",
                block_index=i,
                total_blocks=total_blocks,
                page_index=page_num,
                total_pages=total_pages
            )
            
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
        """Generate translation candidates for a block (optimized for batch-capable backends)."""
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
        
        # Check if backend supports batch candidates
        if self.translator and hasattr(self.translator, 'supports_batch_candidates') and self.translator.supports_batch_candidates:
            # Backend can return N candidates in one call
            from scitran.translation.base import TranslationRequest
            request = TranslationRequest(
                text=user_prompt,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                system_prompt=system_prompt,
                temperature=0.3,
                num_candidates=self.config.num_candidates,
                glossary=self.glossary
            )
            response = self.translator.translate_sync(request)
            candidates = response.translations if response.translations else []
        else:
            # Backend doesn't support batch - generate one at a time
            for i in range(self.config.num_candidates):
                candidate = self._call_translator(system_prompt, user_prompt, temperature=0.3 + i*0.2)
                candidates.append(candidate)
            
        self.stats['total_candidates'] += len(candidates)
        
        return candidates
    
    def _call_translator(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, detect_truncation: bool = True) -> str:
        """Call the translation backend with truncation detection (PHASE 3.1)."""
        if self.translator is None:
            raise ValueError("No translator configured - cannot proceed with translation")
        
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
                num_candidates=1,
                glossary=self.glossary
            )
            
            # Call translator
            response = self.translator.translate_sync(request)
            
            if response.translations:
                translation = response.translations[0]
                finish_reason = response.finish_reasons[0] if response.finish_reasons else None
                
                # PHASE 3.1: Detect truncation
                if detect_truncation and self._is_suspected_partial(user_prompt, translation, finish_reason):
                    logger.warning("Truncation detected, attempting chunk-based translation")
                    # Try chunk-based translation
                    chunks = self._chunk_for_translation(user_prompt)
                    if len(chunks) > 1:
                        translated_chunks = []
                        for chunk in chunks:
                            chunk_request = TranslationRequest(
                                text=chunk,
                                source_lang=self.config.source_lang,
                                target_lang=self.config.target_lang,
                                system_prompt=enhanced_system_prompt,
                                temperature=temperature,
                                num_candidates=1,
                                glossary=self.glossary
                            )
                            chunk_response = self.translator.translate_sync(chunk_request)
                            if chunk_response.translations:
                                translated_chunks.append(chunk_response.translations[0])
                            else:
                                translated_chunks.append(chunk)
                        
                        # Reassemble with same separators as original
                        if '\n\n' in user_prompt:
                            translation = '\n\n'.join(translated_chunks)
                        else:
                            translation = ' '.join(translated_chunks)
                        logger.info(f"Successfully recovered from truncation using {len(chunks)} chunks")
                
                return self._postprocess_translation(translation)
            else:
                raise ValueError("Backend returned empty translation - refusing to use source text")
                
        except ValueError as e:
            # Re-raise ValueErrors (config errors should fail fast)
            raise
        except Exception as e:
            # Re-raise all exceptions (no silent fallback to source text)
            raise RuntimeError(f"Translation failed: {e}") from e
    
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
        """Generate fallback translation when primary fails - must translate or raise exception."""
        text = block.masked_text or block.source_text
        
        # Use fallback translator if available, otherwise create one
        fallback_translator = getattr(self, '_fallback_translator', None)
        if not fallback_translator and self.config.enable_fallback_backend:
            try:
                fallback_translator = self._create_translator(backend_override=self.config.fallback_backend)
                if fallback_translator:
                    self._fallback_translator = fallback_translator
            except Exception as e:
                logger.warning(f"Failed to create fallback translator: {e}")
        
        # If no fallback translator, try primary translator as last resort
        translator_to_use = fallback_translator or self.translator
        if not translator_to_use:
            raise ValueError(f"No translator available for fallback translation of block {block.block_id}")
        
        # Try direct translation without fancy prompts
        from scitran.translation.base import TranslationRequest
        
        request = TranslationRequest(
            text=text,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            temperature=0.3,
            num_candidates=1
        )
        
        try:
            response = translator_to_use.translate_sync(request)
        except Exception as e:
            raise ValueError(f"Fallback translation failed for block {block.block_id}: {e}")
        
        if response.translations and response.translations[0]:
            translation = self._postprocess_translation(response.translations[0])
            # Validate not identical to source
            if self._normalize_text(translation) == self._normalize_text(text):
                raise ValueError(f"Fallback translation identical to source for block {block.block_id}")
            return translation
        
        # No fallback to source text - must fail loudly
        raise ValueError(f"Fallback translation failed for block {block.block_id} - backend returned no translation")

    # ---------- Helpers ----------

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
        # Normalize spacing and quotes FIRST (before stripping control chars)
        cleaned = cleaned.replace("\u00A0", " ")  # nbsp
        cleaned = cleaned.replace("'", "'").replace("'", "'").replace("`", "'").replace("´", "'")
        cleaned = cleaned.replace("\ufffd", "'")  # unknown replacement char to apostrophe
        
        # Preserve newlines but collapse multiple spaces/tabs within lines
        # First, normalize line breaks (CRLF -> LF, CR -> LF)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        
        # Strip control chars EXCEPT newlines (category 'C' but not '\n')
        # We need to preserve newlines for layout
        cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch)[0] != "C" or ch == "\n")
        
        cleaned = self._apply_glossary(cleaned)
        # Collapse multiple spaces/tabs (but not newlines) within each line
        lines = cleaned.split("\n")
        normalized_lines = []
        for line in lines:
            # Collapse spaces/tabs within the line, but preserve single spaces
            normalized_line = re.sub(r"[ \t]+", " ", line)
            # Remove trailing spaces
            normalized_line = normalized_line.rstrip()
            normalized_lines.append(normalized_line)
        cleaned = "\n".join(normalized_lines)
        # Collapse excessive consecutive newlines (max 2)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Remove space before punctuation (but preserve newlines)
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
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
        
        Returns:
            Translator instance
            
        Raises:
            ValueError: If API key is required but not configured
        """
        backend_name = (backend_override or self.config.backend).lower()
        
        # Load API key from config or environment if not provided
        api_key = self.config.api_key
        if not api_key:
            api_key = self._load_api_key(backend_name)

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
                translator = OpenAIBackend(api_key=api_key, model=self.config.model_name)
                if not translator.is_available():
                    raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable or use 'scitrans keys set --backend openai --key <key>'")
                return translator
            elif backend_name == "anthropic":
                from scitran.translation.backends.anthropic_backend import AnthropicBackend
                translator = AnthropicBackend(api_key=api_key, model=self.config.model_name)
                if not translator.is_available():
                    raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable or use 'scitrans keys set --backend anthropic --key <key>'")
                return translator
            elif backend_name == "deepseek":
                from scitran.translation.backends.deepseek_backend import DeepSeekBackend
                translator = DeepSeekBackend(api_key=api_key, model=self.config.model_name)
                if not translator.is_available():
                    raise ValueError("DeepSeek API key not configured. Set DEEPSEEK_API_KEY environment variable or use 'scitrans keys set --backend deepseek --key <key>'")
                return translator
            elif backend_name == "local":
                from scitran.translation.backends.local_backend import LocalBackend
                return LocalBackend()
            elif backend_name == "libre":
                from scitran.translation.backends.libre_backend import LibreTranslateBackend
                return LibreTranslateBackend(api_key=api_key, model=self.config.model_name)
            elif backend_name == "argos":
                from scitran.translation.backends.argos_backend import ArgosBackend
                return ArgosBackend()
            else:
                logger.warning(f"Unknown backend: {backend_name}, using cascade")
                from scitran.translation.backends.cascade_backend import CascadeBackend
                return CascadeBackend()
        except ImportError as e:
            import traceback
            logger.error(f"Could not import backend {backend_name}: {e}")
            return None
        except ValueError as e:
            # Re-raise ValueError (API key errors) - these should fail fast
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error creating backend {backend_name}: {e}")
            return None
    
    def _load_api_key(self, backend_name: str) -> Optional[str]:
        """Load API key from environment or config file.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            API key if found, None otherwise
        """
        import os
        from pathlib import Path
        
        # Environment variable mappings
        env_mappings = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        
        # Check environment variable first
        if backend_name in env_mappings:
            api_key = os.getenv(env_mappings[backend_name])
            if api_key:
                return api_key
        
        # Check config file
        config_path = Path.home() / ".scitrans" / "config.yaml"
        if config_path.exists():
            try:
                from scitran.utils.config_loader import load_config
                config = load_config(str(config_path))
                api_keys = config.get("api_keys", {})
                if backend_name in api_keys:
                    return api_keys[backend_name]
            except Exception:
                pass
        
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
    
    def _report_progress(
        self, 
        progress: float, 
        message: str,
        block_index: Optional[int] = None,
        total_blocks: Optional[int] = None,
        page_index: Optional[int] = None,
        total_pages: Optional[int] = None
    ):
        """
        Report progress to callback if provided.
        
        Enhanced with per-block and per-page granularity.
        
        Args:
            progress: Overall progress (0.0-1.0)
            message: Progress message
            block_index: Current block index (for per-block progress)
            total_blocks: Total number of blocks (for per-block progress)
            page_index: Current page index (for per-page progress)
            total_pages: Total number of pages (for per-page progress)
        """
        # Build enhanced message with granular info
        enhanced_message = message
        if block_index is not None and total_blocks is not None:
            enhanced_message += f" (Block {block_index + 1}/{total_blocks})"
        if page_index is not None and total_pages is not None:
            enhanced_message += f" (Page {page_index + 1}/{total_pages})"
        
        if self.progress_callback:
            # Pass additional context if callback supports it
            if hasattr(self.progress_callback, '__code__'):
                # Check if callback accepts more arguments
                import inspect
                sig = inspect.signature(self.progress_callback)
                if len(sig.parameters) > 2:
                    # Enhanced callback with granular info
                    self.progress_callback(
                        progress, 
                        enhanced_message,
                        block_index=block_index,
                        total_blocks=total_blocks,
                        page_index=page_index,
                        total_pages=total_pages
                    )
                else:
                    # Standard callback
                    self.progress_callback(progress, enhanced_message)
            else:
                self.progress_callback(progress, enhanced_message)
        
        logger.info(f"[{progress:.0%}] {enhanced_message}")
        
    def _log_debug_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured debug event to JSONL file (PHASE 4.1)."""
        if not self._debug_log_file:
            return
        
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                **data
            }
            self._debug_log_file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self._debug_log_file.flush()
        except Exception as e:
            logger.debug(f"Failed to write debug log: {e}")
    
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
    
    def __del__(self):
        """Cleanup debug log file."""
        if self._debug_log_file:
            try:
                self._debug_log_file.close()
            except:
                pass
    
    def get_run_summary(self, result: TranslationResult) -> Dict[str, Any]:
        """Generate comprehensive run summary (PHASE 4.2).
        
        Args:
            result: TranslationResult from translate_document
        
        Returns:
            Dictionary with summary statistics
        """
        document = result.document
        total_blocks = len(document.all_blocks)
        translatable_blocks = len(document.translatable_blocks)
        
        # Count blocks by type
        blocks_by_type = {}
        for block in document.all_blocks:
            block_type = block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type)
            blocks_by_type[block_type] = blocks_by_type.get(block_type, 0) + 1
        
        # Calculate speed metrics
        blocks_per_sec = result.blocks_translated / result.duration if result.duration > 0 else 0
        
        summary = {
            "success": result.success,
            "duration_seconds": round(result.duration, 2),
            "backend": result.backend_used,
            
            # Block statistics
            "blocks": {
                "total": total_blocks,
                "translatable": translatable_blocks,
                "translated_ok": result.blocks_translated,
                "failed": result.blocks_failed,
                "by_type": blocks_by_type,
            },
            
            # Coverage
            "coverage": {
                "ratio": result.coverage,
                "percentage": f"{result.coverage * 100:.1f}%",
            },
            
            # Performance
            "performance": {
                "blocks_per_second": round(blocks_per_sec, 2),
                "cache_hits": self.stats.get('cache_hits', 0) + self.stats.get('batch_cache_hits', 0),
                "cache_misses": self.stats.get('cache_misses', 0),
            },
            
            # Quality metrics
            "quality": {
                "bleu": result.bleu_score,
                "chrf": result.chrf_score,
                "glossary_adherence": result.glossary_adherence,
                "layout_preservation": result.layout_preservation,
            },
            
            # Masking
            "masking": {
                "masks_applied": result.masks_applied,
                "masks_restored": result.masks_restored,
                "violations": self.stats.get('mask_violations', 0),
            },
        }
        
        # Add failure report if present
        if result.failure_report:
            summary["failures"] = result.failure_report
        
        return summary
    
    def print_run_summary(self, result: TranslationResult):
        """Print human-readable run summary (PHASE 4.2)."""
        summary = self.get_run_summary(result)
        
        print("\n" + "="*60)
        print("TRANSLATION RUN SUMMARY")
        print("="*60)
        print(f"Status: {'✓ SUCCESS' if summary['success'] else '✗ FAILED'}")
        print(f"Duration: {summary['duration_seconds']}s")
        print(f"Backend: {summary['backend']}")
        print()
        
        print("BLOCKS:")
        print(f"  Total: {summary['blocks']['total']}")
        print(f"  Translatable: {summary['blocks']['translatable']}")
        print(f"  Translated: {summary['blocks']['translated_ok']}")
        print(f"  Failed: {summary['blocks']['failed']}")
        print()
        
        print("COVERAGE:")
        print(f"  {summary['coverage']['percentage']}")
        print()
        
        print("PERFORMANCE:")
        print(f"  Speed: {summary['performance']['blocks_per_second']} blocks/sec")
        print(f"  Cache hits: {summary['performance']['cache_hits']}")
        print()
        
        if summary.get('failures'):
            print("FAILURES:")
            print(f"  {len(summary['failures'].get('failures', []))} blocks failed")
        
        print("="*60 + "\n")

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
    
    # Glossary settings
    enable_glossary: bool = True
    glossary_path: Optional[Path] = None
    domain: str = "scientific"  # scientific, medical, legal, technical
    
    # Quality settings
    num_candidates: int = 3  # Number of candidates to generate
    enable_reranking: bool = True
    reranking_strategy: ScoringStrategy = ScoringStrategy.HYBRID
    quality_threshold: float = 0.7  # Minimum acceptable quality score
    
    # Prompt settings
    prompt_template: str = "scientific_expert"
    optimize_prompts: bool = True
    prompt_optimization_rounds: int = 3
    
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
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
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
            'quality_failures': 0
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
        
        # Initialize result
        result = TranslationResult(
            document=document,
            backend_used=self.config.backend
        )
        
        try:
            # Setup
            self._setup_translation(document)
            
            # Phase 1: Apply masking (Innovation #1)
            if self.config.enable_masking:
                self._report_progress(0.1, "Applying content masking...")
                self.masking_engine.reset()
                document = self.masking_engine.mask_document(document)
                result.masks_applied = document.stats.get('total_masks', 0)
            
            # Phase 2: Translate blocks with context (Innovation #2)
            self._report_progress(0.2, "Translating document...")
            self._translate_all_blocks(document)
            
            # Phase 3: Restore masks with validation
            if self.config.enable_masking:
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
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.success = False
            result.error = str(e)
            
        result.duration = time.time() - start_time
        
        return result
    
    def _setup_translation(self, document: Document):
        """Initialize translation components."""
        # Load glossary
        if self.config.enable_glossary:
            self.glossary = self._load_glossary(document)
            
        # Initialize translator
        self.translator = self._create_translator()
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}
        
    def _translate_all_blocks(self, document: Document):
        """Translate all blocks in the document."""
        total_blocks = len(document.translatable_blocks)
        
        for i, block in enumerate(document.translatable_blocks):
            progress = 0.2 + (0.6 * i / max(total_blocks, 1))
            self._report_progress(progress, f"Translating block {i+1}/{total_blocks}")
            
            try:
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
        
        # Get optimized prompt
        if self.prompt_optimizer:
            system_prompt, user_prompt = self.prompt_optimizer.generate_prompt(
                block=block,
                template_name=self.config.prompt_template,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                glossary_terms=self.glossary,
                context=context
            )
        else:
            # Use default prompt
            system_prompt = f"Translate from {self.config.source_lang} to {self.config.target_lang}"
            user_prompt = block.masked_text or block.source_text
            
        # Generate candidates (mock for now - would call actual translator)
        for i in range(self.config.num_candidates):
            # In real implementation, would call translator with temperature variation
            candidate = self._call_translator(system_prompt, user_prompt, temperature=0.3 + i*0.2)
            candidates.append(candidate)
            
        self.stats['total_candidates'] += len(candidates)
        
        return candidates
    
    def _call_translator(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Call the translation backend (mock implementation)."""
        # This would call the actual translator
        # For now, return a mock translation
        return f"[Translation of: {user_prompt[:50]}...]"
    
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
        """Generate fallback translation when primary fails."""
        # Try simpler prompt
        simple_prompt = f"Translate: {block.source_text[:100]}"
        
        # In real implementation, would try different strategy
        return f"[Fallback translation of block {block.block_id}]"
    
    def _validate_translation(self, document: Document) -> Dict[str, float]:
        """Validate overall translation quality."""
        metrics = {}
        
        # Calculate glossary adherence
        if self.glossary:
            correct = 0
            total = 0
            for block in document.translatable_blocks:
                if block.translated_text:
                    for term, translation in self.glossary.items():
                        if term.lower() in block.source_text.lower():
                            total += 1
                            if translation.lower() in block.translated_text.lower():
                                correct += 1
            metrics['glossary_adherence'] = correct / total if total > 0 else 1.0
            
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
    
    def _load_glossary(self, document: Document) -> Dict[str, str]:
        """Load glossary for the document."""
        glossary = {}
        
        # Start with document-specific glossary
        if document.glossary_terms:
            glossary.update(document.glossary_terms)
            
        # Add domain-specific glossary
        # In real implementation, would load from file based on domain
        if self.config.domain == "scientific":
            glossary.update({
                "machine learning": "apprentissage automatique",
                "deep learning": "apprentissage profond",
                "neural network": "réseau de neurones",
                "natural language processing": "traitement du langage naturel",
                # ... more terms
            })
            
        return glossary
    
    def _create_translator(self):
        """Create translator instance based on backend."""
        # In real implementation, would create actual translator
        # based on self.config.backend
        return None
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if provided."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        logger.info(f"[{progress:.0%}] {message}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        stats = dict(self.stats)
        
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

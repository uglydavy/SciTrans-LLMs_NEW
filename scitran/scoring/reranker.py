"""
Advanced reranking and scoring system for SciTrans-LLMs NEW.

This module implements multi-dimensional quality assessment and candidate
selection for optimal translation quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import re
import math
from collections import Counter
import logging

from scitran.core.models import Block, MaskInfo

# Optional numpy support - fallback to pure Python if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    
    # Create a simple namespace to provide numpy-like functions
    class NumpyFallback:
        @staticmethod
        def mean(values):
            if not values:
                return 0.0
            return sum(values) / len(values)
        
        @staticmethod
        def std(values):
            if len(values) < 2:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
    
    np = NumpyFallback()

logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """A dimension of translation quality."""
    name: str
    weight: float
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        """Get weighted contribution to overall score."""
        return self.score * self.weight


@dataclass
class TranslationCandidate:
    """A translation candidate with quality scores."""
    text: str
    source_text: str
    candidate_id: int
    
    # Individual dimension scores
    dimensions: Dict[str, QualityDimension] = field(default_factory=dict)
    
    # Overall score
    total_score: float = 0.0
    
    # Metadata
    backend: Optional[str] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total_score(self):
        """Calculate total score from dimensions."""
        self.total_score = sum(d.weighted_score for d in self.dimensions.values())
        return self.total_score


class ScoringStrategy(Enum):
    """Different scoring strategies."""
    HEURISTIC = "heuristic"      # Rule-based scoring
    LLM_BASED = "llm_based"       # LLM-based scoring
    HYBRID = "hybrid"             # Combination of both
    REFERENCE = "reference"       # Reference-based (BLEU, etc.)


class MultiDimensionalScorer:
    """
    Scores translation candidates across multiple quality dimensions.
    
    Key dimensions:
    1. Fluency - Natural language quality
    2. Adequacy - Semantic preservation
    3. Terminology - Glossary adherence
    4. Format - Structure and placeholder preservation
    5. Consistency - Document-level coherence
    """
    
    def __init__(self, 
                 strategy: ScoringStrategy = ScoringStrategy.HYBRID,
                 custom_weights: Optional[Dict[str, float]] = None):
        self.strategy = strategy
        self.weights = self._get_default_weights()
        if custom_weights:
            self.weights.update(custom_weights)
            
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default dimension weights."""
        return {
            'fluency': 0.20,
            'adequacy': 0.30,
            'terminology': 0.25,
            'format': 0.20,
            'consistency': 0.05
        }
    
    def score_candidates(self,
                        candidates: List[str],
                        source_text: str,
                        masks: List[MaskInfo] = None,
                        glossary: Dict[str, str] = None,
                        context: List[Tuple[str, str]] = None,
                        **kwargs) -> List[TranslationCandidate]:
        """
        Score all candidates and return sorted list.
        
        Args:
            candidates: List of translation candidates
            source_text: Original source text
            masks: Mask information for format checking
            glossary: Glossary terms for terminology scoring
            context: Previous translations for consistency
            
        Returns:
            List of scored candidates, sorted by total score (best first)
        """
        scored_candidates = []
        
        for i, candidate_text in enumerate(candidates):
            candidate = TranslationCandidate(
                text=candidate_text,
                source_text=source_text,
                candidate_id=i
            )
            
            # Score each dimension
            candidate.dimensions['fluency'] = self._score_fluency(
                candidate_text, self.weights['fluency']
            )
            
            candidate.dimensions['adequacy'] = self._score_adequacy(
                source_text, candidate_text, self.weights['adequacy']
            )
            
            candidate.dimensions['terminology'] = self._score_terminology(
                candidate_text, glossary or {}, self.weights['terminology']
            )
            
            candidate.dimensions['format'] = self._score_format(
                candidate_text, masks or [], self.weights['format']
            )
            
            candidate.dimensions['consistency'] = self._score_consistency(
                candidate_text, context or [], self.weights['consistency']
            )
            
            # Calculate total
            candidate.calculate_total_score()
            scored_candidates.append(candidate)
            
        # Sort by total score (descending)
        scored_candidates.sort(key=lambda c: c.total_score, reverse=True)
        
        return scored_candidates
    
    def _score_fluency(self, text: str, weight: float) -> QualityDimension:
        """
        Score translation fluency (grammatical correctness, naturalness).
        
        Heuristic approach:
        - Check for repeated words
        - Sentence length distribution
        - Punctuation balance
        - Capital letter usage
        """
        dimension = QualityDimension(name="fluency", weight=weight)
        
        # Basic text statistics
        sentences = text.split('. ')
        words = text.split()
        
        if not words:
            dimension.score = 0.0
            return dimension
            
        scores = []
        
        # 1. Repeated words (lower is better)
        word_freq = Counter(word.lower() for word in words if len(word) > 3)
        max_freq = max(word_freq.values()) if word_freq else 1
        repetition_score = 1.0 - min(max_freq / len(words), 0.3) / 0.3
        scores.append(repetition_score)
        dimension.details['repetition'] = repetition_score
        
        # 2. Sentence length variety (higher is better)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            length_std = np.std(lengths)
            length_variety = min(length_std / 10, 1.0)  # Normalize to 0-1
            scores.append(length_variety)
            dimension.details['length_variety'] = length_variety
        
        # 3. Punctuation balance
        punct_count = sum(1 for c in text if c in '.,;:!?')
        punct_ratio = punct_count / len(words)
        punct_score = 1.0 - abs(punct_ratio - 0.15) / 0.15  # Optimal ~15%
        punct_score = max(0, punct_score)
        scores.append(punct_score)
        dimension.details['punctuation'] = punct_score
        
        # 4. Capitalization (proper nouns, sentence starts)
        capital_count = sum(1 for word in words if word[0].isupper())
        capital_ratio = capital_count / len(words)
        capital_score = 1.0 - abs(capital_ratio - 0.1) / 0.1  # Optimal ~10%
        capital_score = max(0, capital_score)
        scores.append(capital_score)
        dimension.details['capitalization'] = capital_score
        
        # Average all scores
        dimension.score = sum(scores) / len(scores) if scores else 0.0
        
        return dimension
    
    def _score_adequacy(self, source: str, translation: str, weight: float) -> QualityDimension:
        """
        Score semantic adequacy (meaning preservation).
        
        Heuristic approach:
        - Length ratio
        - Number preservation
        - Named entity preservation
        - Key term coverage
        """
        dimension = QualityDimension(name="adequacy", weight=weight)
        
        scores = []
        
        # 1. Length ratio (should be similar)
        source_len = len(source.split())
        trans_len = len(translation.split())
        if source_len > 0:
            ratio = trans_len / source_len
            # Optimal ratio depends on language pair, but generally 0.8-1.2
            length_score = 1.0 - min(abs(ratio - 1.0), 0.5) / 0.5
            scores.append(length_score)
            dimension.details['length_ratio'] = ratio
        
        # 2. Number preservation
        source_nums = re.findall(r'\b\d+(?:\.\d+)?\b', source)
        trans_nums = re.findall(r'\b\d+(?:\.\d+)?\b', translation)
        if source_nums:
            preserved = sum(1 for num in source_nums if num in trans_nums)
            num_score = preserved / len(source_nums)
            scores.append(num_score)
            dimension.details['number_preservation'] = num_score
        
        # 3. Capitalized word preservation (likely named entities)
        source_caps = set(word for word in source.split() if word[0].isupper())
        trans_caps = set(word for word in translation.split() if word[0].isupper())
        if source_caps:
            # Some may be translated, so check for partial matches too
            preserved_caps = len(source_caps.intersection(trans_caps))
            cap_score = min(preserved_caps / len(source_caps), 1.0)
            scores.append(cap_score)
            dimension.details['entity_preservation'] = cap_score
        
        # 4. Parentheses and brackets preservation
        source_parens = source.count('(') + source.count('[')
        trans_parens = translation.count('(') + translation.count('[')
        if source_parens > 0:
            paren_score = min(trans_parens / source_parens, 1.0)
            scores.append(paren_score)
            dimension.details['structure_preservation'] = paren_score
        
        dimension.score = sum(scores) / len(scores) if scores else 0.5
        
        return dimension
    
    def _score_terminology(self, text: str, glossary: Dict[str, str], weight: float) -> QualityDimension:
        """
        Score glossary term adherence.
        
        Checks if glossary terms are correctly translated.
        """
        dimension = QualityDimension(name="terminology", weight=weight)
        
        if not glossary:
            dimension.score = 1.0  # No glossary to check
            return dimension
            
        found_terms = 0
        correct_terms = 0
        
        text_lower = text.lower()
        
        for source_term, target_term in glossary.items():
            # Check if the target term appears in translation
            if target_term.lower() in text_lower:
                found_terms += 1
                correct_terms += 1
            # Check if source term wrongly appears (untranslated)
            elif source_term.lower() in text_lower:
                found_terms += 1
                # This is incorrect - term not translated
                
        if found_terms > 0:
            dimension.score = correct_terms / found_terms
        else:
            dimension.score = 1.0  # No terms to check
            
        dimension.details['terms_found'] = found_terms
        dimension.details['terms_correct'] = correct_terms
        
        return dimension
    
    def _score_format(self, text: str, masks: List[MaskInfo], weight: float) -> QualityDimension:
        """
        Score format preservation (placeholders, structure).
        
        Critical for Innovation #1: Checks if all masks are preserved.
        """
        dimension = QualityDimension(name="format", weight=weight)
        
        if not masks:
            dimension.score = 1.0
            return dimension
            
        preserved = 0
        total = len(masks)
        
        for mask in masks:
            if mask.placeholder in text:
                preserved += 1
                
        dimension.score = preserved / total if total > 0 else 1.0
        dimension.details['masks_preserved'] = preserved
        dimension.details['masks_total'] = total
        
        # Additional format checks
        format_scores = [dimension.score]  # Mask preservation is most important
        
        # Check list marker preservation
        source_lists = len(re.findall(r'^\s*[\d•\-\*]\s', text, re.MULTILINE))
        if source_lists > 0:
            list_score = min(source_lists / max(source_lists, 1), 1.0)
            format_scores.append(list_score)
            
        # Check line break preservation (paragraphs)
        source_breaks = text.count('\n\n')
        if source_breaks > 0:
            break_score = 1.0  # Simple check for now
            format_scores.append(break_score)
            
        dimension.score = sum(format_scores) / len(format_scores)
        
        return dimension
    
    def _score_consistency(self, text: str, context: List[Tuple[str, str]], weight: float) -> QualityDimension:
        """
        Score consistency with previous translations.
        
        Checks for consistent terminology and style.
        """
        dimension = QualityDimension(name="consistency", weight=weight)
        
        if not context:
            dimension.score = 1.0
            return dimension
            
        # Extract recurring terms from context
        context_terms = Counter()
        for _, prev_trans in context:
            words = prev_trans.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on content words
                    context_terms[word] += 1
                    
        # Find frequent terms (appear in multiple previous translations)
        frequent_terms = [term for term, count in context_terms.items() if count >= 2]
        
        if not frequent_terms:
            dimension.score = 1.0
            return dimension
            
        # Check how many frequent terms appear in current translation
        text_lower = text.lower()
        found = sum(1 for term in frequent_terms if term in text_lower)
        
        dimension.score = min(found / len(frequent_terms), 1.0)
        dimension.details['consistent_terms'] = found
        dimension.details['expected_terms'] = len(frequent_terms)
        
        return dimension


class AdvancedReranker:
    """
    Advanced reranking system with multiple strategies and learning capability.
    
    This implements Innovation #2 by selecting the best translation
    from multiple candidates using sophisticated scoring.
    """
    
    def __init__(self,
                 scorer: Optional[MultiDimensionalScorer] = None,
                 strategy: ScoringStrategy = ScoringStrategy.HYBRID):
        self.scorer = scorer or MultiDimensionalScorer(strategy=strategy)
        self.reranking_history: List[Dict[str, Any]] = []
        
    def rerank(self,
               candidates: List[str],
               block: Block,
               glossary: Dict[str, str] = None,
               context: List[Tuple[str, str]] = None,
               require_all_masks: bool = True,
               **kwargs) -> Tuple[str, List[TranslationCandidate]]:
        """
        Rerank candidates and return best translation.
        
        Args:
            candidates: List of translation candidates
            block: Source block being translated
            glossary: Glossary terms
            context: Previous translations
            require_all_masks: Whether to require all masks preserved
            
        Returns:
            Tuple of (best_translation, all_scored_candidates)
        """
        if not candidates:
            return "", []
            
        if len(candidates) == 1:
            return candidates[0], []
            
        # Score all candidates
        scored = self.scorer.score_candidates(
            candidates=candidates,
            source_text=block.source_text,
            masks=block.masks,
            glossary=glossary,
            context=context,
            **kwargs
        )
        
        # Apply hard constraints if needed
        if require_all_masks and block.masks:
            # Filter out candidates missing masks
            valid_scored = [
                c for c in scored 
                if c.dimensions['format'].details.get('masks_preserved', 0) == 
                   c.dimensions['format'].details.get('masks_total', 0)
            ]
            
            if valid_scored:
                scored = valid_scored
            else:
                # No candidate preserves all masks - log warning
                logger.warning(f"Block {block.block_id}: No candidate preserves all masks")
        
        # Record reranking decision
        self.reranking_history.append({
            'block_id': block.block_id,
            'num_candidates': len(candidates),
            'best_score': scored[0].total_score if scored else 0,
            'score_distribution': [c.total_score for c in scored],
            'selected_idx': scored[0].candidate_id if scored else 0
        })
        
        # Update block metadata with scores
        if block.metadata:
            # Ensure metadata is TranslationMetadata object, not dict
            from scitran.core.models import TranslationMetadata
            from datetime import datetime
            
            if isinstance(block.metadata, dict):
                # Convert dict to TranslationMetadata
                block.metadata = TranslationMetadata(
                    backend=block.metadata.get('backend', 'unknown'),
                    timestamp=block.metadata.get('timestamp', datetime.now()),
                    duration=block.metadata.get('duration', 0.0),
                    confidence=block.metadata.get('confidence', 0.0),
                    glossary_terms_used=set(block.metadata.get('glossary_terms_used', [])),
                    context_used=block.metadata.get('context_used', False),
                    candidates_generated=block.metadata.get('candidates_generated', 1),
                    reranking_applied=block.metadata.get('reranking_applied', False)
                )
            
            block.metadata.candidates_generated = len(candidates)
            block.metadata.reranking_applied = True
            block.metadata.confidence = scored[0].total_score if scored else 0.0
        
        # Return best translation and all scored candidates
        best_translation = scored[0].text if scored else candidates[0]
        return best_translation, scored
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get statistics about reranking performance."""
        if not self.reranking_history:
            return {}
            
        scores = [h['best_score'] for h in self.reranking_history]
        num_candidates = [h['num_candidates'] for h in self.reranking_history]
        
        return {
            'total_rerankings': len(self.reranking_history),
            'avg_best_score': np.mean(scores),
            'std_best_score': np.std(scores),
            'avg_num_candidates': np.mean(num_candidates),
            'score_improvement': self._calculate_improvement()
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate average improvement from reranking."""
        improvements = []
        
        for record in self.reranking_history:
            scores = record['score_distribution']
            if len(scores) > 1:
                # Compare best (after reranking) to first (before reranking)
                first_score = scores[record['selected_idx']]
                # Find original first candidate score
                original_first = next(
                    (s for i, s in enumerate(scores) if record['selected_idx'] == 0),
                    scores[0]
                )
                improvement = first_score - original_first
                improvements.append(improvement)
                
        return np.mean(improvements) if improvements else 0.0
    
    def adapt_weights(self, feedback: Dict[str, float]):
        """
        Adapt scoring weights based on human feedback.
        
        Args:
            feedback: Dictionary of dimension -> importance (0-1)
        """
        # Normalize feedback to sum to 1
        total = sum(feedback.values())
        if total > 0:
            normalized = {k: v/total for k, v in feedback.items()}
            
            # Blend with current weights (learning rate = 0.1)
            for dim, importance in normalized.items():
                if dim in self.scorer.weights:
                    current = self.scorer.weights[dim]
                    self.scorer.weights[dim] = 0.9 * current + 0.1 * importance
                    
        # Re-normalize weights
        total = sum(self.scorer.weights.values())
        self.scorer.weights = {k: v/total for k, v in self.scorer.weights.items()}

"""
Training algorithm for improving SciTrans system based on evaluation metrics.

This module implements:
1. Failure analysis
2. Feature extraction
3. Improvement generation
4. Training loop
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example from evaluation."""
    source_text: str
    reference_translation: Optional[str]
    system_translation: Optional[str]
    failure_type: str  # masking|quality|detection|style|coverage
    block_id: str
    block_type: str
    features: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str


@dataclass
class FailurePattern:
    """Pattern identified from failures."""
    pattern_type: str
    description: str
    frequency: int
    examples: List[str]
    suggested_fix: str


class FailureAnalyzer:
    """Analyzes failures and extracts patterns."""
    
    def __init__(self):
        self.failures: List[TrainingExample] = []
        self.patterns: List[FailurePattern] = []
    
    def analyze_evaluation_results(
        self,
        evaluation_results: List[Dict[str, Any]],
        document: Any  # Document object
    ) -> List[TrainingExample]:
        """
        Analyze evaluation results and extract training examples.
        
        Args:
            evaluation_results: List of evaluation metrics per PDF
            document: Document object with blocks
            
        Returns:
            List of training examples
        """
        examples = []
        
        for result in evaluation_results:
            if not result.get('success', False):
                continue
            
            # Analyze masking failures
            masking_failures = self._analyze_masking_failures(result, document)
            examples.extend(masking_failures)
            
            # Analyze translation quality failures
            quality_failures = self._analyze_quality_failures(result, document)
            examples.extend(quality_failures)
            
            # Analyze coverage failures
            coverage_failures = self._analyze_coverage_failures(result, document)
            examples.extend(coverage_failures)
        
        self.failures.extend(examples)
        return examples
    
    def _analyze_masking_failures(
        self,
        result: Dict[str, Any],
        document: Any
    ) -> List[TrainingExample]:
        """Analyze masking-related failures."""
        examples = []
        masking_metrics = result.get('masking', {})
        
        if masking_metrics.get('loss_rate', 0) > 0 or masking_metrics.get('corruption_rate', 0) > 0:
            # Find blocks with masking failures
            for block in document.translatable_blocks:
                if not block.masks or not block.translated_text:
                    continue
                
                # Check for lost or corrupted masks
                for mask in block.masks:
                    if mask.placeholder not in block.translated_text:
                        if mask.original in block.translated_text:
                            # Corrupted: original appeared
                            failure_type = "masking_corrupted"
                        else:
                            # Lost: neither placeholder nor original
                            failure_type = "masking_lost"
                        
                        features = self._extract_features(block, mask)
                        examples.append(TrainingExample(
                            source_text=block.source_text,
                            reference_translation=None,
                            system_translation=block.translated_text,
                            failure_type=failure_type,
                            block_id=block.block_id,
                            block_type=block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
                            features=features,
                            metrics={"mask_type": mask.mask_type.value if hasattr(mask.mask_type, 'value') else str(mask.mask_type)},
                            timestamp=datetime.now().isoformat()
                        ))
        
        return examples
    
    def _analyze_quality_failures(
        self,
        result: Dict[str, Any],
        document: Any
    ) -> List[TrainingExample]:
        """Analyze translation quality failures."""
        examples = []
        
        # Check BLEU/chrF scores
        bleu = result.get('bleu')
        chrf = result.get('chrf')
        
        # Threshold for quality failure
        quality_threshold = 0.5
        
        if bleu is not None and bleu < quality_threshold:
            # Find blocks with low quality
            for block in document.translatable_blocks:
                if block.translated_text:
                    # Simple heuristic: if block is long and translation is similar length, might be OK
                    # But if very different, likely poor quality
                    length_ratio = len(block.translated_text) / len(block.source_text) if block.source_text else 1.0
                    
                    if length_ratio < 0.3 or length_ratio > 3.0:
                        features = self._extract_features(block)
                        examples.append(TrainingExample(
                            source_text=block.source_text,
                            reference_translation=None,
                            system_translation=block.translated_text,
                            failure_type="quality",
                            block_id=block.block_id,
                            block_type=block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
                            features=features,
                            metrics={"bleu": bleu, "chrf": chrf, "length_ratio": length_ratio},
                            timestamp=datetime.now().isoformat()
                        ))
        
        return examples
    
    def _analyze_coverage_failures(
        self,
        result: Dict[str, Any],
        document: Any
    ) -> List[TrainingExample]:
        """Analyze coverage failures (untranslated blocks)."""
        examples = []
        
        coverage_metrics = result.get('coverage', {})
        failed_block_ids = coverage_metrics.get('failed_block_ids', [])
        
        for block_id in failed_block_ids:
            block = document.get_block_by_id(block_id)
            if block:
                features = self._extract_features(block)
                examples.append(TrainingExample(
                    source_text=block.source_text,
                    reference_translation=None,
                    system_translation=None,
                    failure_type="coverage",
                    block_id=block.block_id,
                    block_type=block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
                    features=features,
                    metrics={},
                    timestamp=datetime.now().isoformat()
                ))
        
        return examples
    
    def _extract_features(self, block: Any, mask: Optional[Any] = None) -> Dict[str, Any]:
        """Extract features from a block."""
        features = {
            "text_length": len(block.source_text),
            "word_count": len(block.source_text.split()),
            "has_numbers": bool(re.search(r'\d', block.source_text)),
            "has_math": bool(re.search(r'[=+\-*/^]', block.source_text)),
            "has_urls": bool(re.search(r'https?://', block.source_text)),
            "has_email": bool(re.search(r'@', block.source_text)),
            "block_type": block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type),
        }
        
        if mask:
            features["mask_type"] = mask.mask_type.value if hasattr(mask.mask_type, 'value') else str(mask.mask_type)
            features["mask_original_length"] = len(mask.original)
            features["mask_placeholder_length"] = len(mask.placeholder)
        
        return features
    
    def identify_patterns(self) -> List[FailurePattern]:
        """Identify common patterns in failures."""
        patterns = []
        
        # Group failures by type
        failures_by_type = {}
        for failure in self.failures:
            failure_type = failure.failure_type
            if failure_type not in failures_by_type:
                failures_by_type[failure_type] = []
            failures_by_type[failure_type].append(failure)
        
        # Analyze each failure type
        for failure_type, failures in failures_by_type.items():
            if len(failures) < 3:  # Need at least 3 examples
                continue
            
            # Extract common features
            common_features = self._find_common_features(failures)
            
            pattern = FailurePattern(
                pattern_type=failure_type,
                description=self._describe_pattern(failures, common_features),
                frequency=len(failures),
                examples=[f.source_text[:100] for f in failures[:5]],
                suggested_fix=self._suggest_fix(failure_type, common_features)
            )
            patterns.append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _find_common_features(self, failures: List[TrainingExample]) -> Dict[str, Any]:
        """Find common features across failures."""
        if not failures:
            return {}
        
        # Count feature occurrences
        feature_counts = {}
        for failure in failures:
            for key, value in failure.features.items():
                if key not in feature_counts:
                    feature_counts[key] = {}
                if value not in feature_counts[key]:
                    feature_counts[key][value] = 0
                feature_counts[key][value] += 1
        
        # Find most common values
        common = {}
        for key, counts in feature_counts.items():
            if counts:
                most_common = max(counts.items(), key=lambda x: x[1])
                if most_common[1] >= len(failures) * 0.5:  # At least 50% share
                    common[key] = most_common[0]
        
        return common
    
    def _describe_pattern(
        self,
        failures: List[TrainingExample],
        common_features: Dict[str, Any]
    ) -> str:
        """Generate human-readable pattern description."""
        desc = f"{len(failures)} failures of type {failures[0].failure_type}"
        if common_features:
            desc += f" with common features: {common_features}"
        return desc
    
    def _suggest_fix(self, failure_type: str, common_features: Dict[str, Any]) -> str:
        """Suggest a fix based on failure type and features."""
        suggestions = {
            "masking_lost": "Improve masking rules for this content type",
            "masking_corrupted": "Strengthen placeholder preservation in prompts",
            "quality": "Adjust translation prompts or use different backend",
            "coverage": "Improve retry logic or fallback mechanisms",
        }
        
        base_suggestion = suggestions.get(failure_type, "Review and improve handling")
        
        if common_features:
            if "mask_type" in common_features:
                base_suggestion += f" (especially for {common_features['mask_type']})"
            if "block_type" in common_features:
                base_suggestion += f" (for {common_features['block_type']} blocks)"
        
        return base_suggestion


class PromptOptimizer:
    """Optimizes prompts based on failure patterns."""
    
    def __init__(self):
        self.prompt_variations = []
        self.best_prompts = {}
    
    def generate_variations(
        self,
        base_prompt: str,
        failure_patterns: List[FailurePattern]
    ) -> List[str]:
        """Generate prompt variations to address failure patterns."""
        variations = [base_prompt]  # Keep original
        
        for pattern in failure_patterns:
            if pattern.pattern_type == "masking_lost" or pattern.pattern_type == "masking_corrupted":
                # Add stronger placeholder preservation
                variation = base_prompt + "\n\nCRITICAL: Preserve all placeholder tokens EXACTLY as they appear. Do not translate or modify placeholders."
                variations.append(variation)
            
            elif pattern.pattern_type == "quality":
                # Add quality emphasis
                variation = base_prompt + "\n\nIMPORTANT: Ensure accurate, natural translation that preserves meaning and style."
                variations.append(variation)
        
        return variations
    
    def select_best_prompt(
        self,
        variations: List[str],
        validation_results: List[Dict[str, Any]]
    ) -> str:
        """Select best-performing prompt based on validation results."""
        if not validation_results:
            return variations[0]
        
        # Score each variation
        scores = {}
        for i, variation in enumerate(variations):
            # Find results for this variation
            var_results = [r for r in validation_results if r.get('prompt_index') == i]
            if var_results:
                # Average BLEU/chrF
                avg_bleu = sum(r.get('bleu', 0) or 0 for r in var_results) / len(var_results)
                avg_chrf = sum(r.get('chrf', 0) or 0 for r in var_results) / len(var_results)
                scores[i] = (avg_bleu + avg_chrf) / 2
        
        if scores:
            best_idx = max(scores.items(), key=lambda x: x[1])[0]
            return variations[best_idx]
        
        return variations[0]


class TrainingLoop:
    """Main training loop for system improvement."""
    
    def __init__(
        self,
        test_pdfs_dir: Path,
        reference_dir: Optional[Path] = None,
        output_dir: Path = Path("training_results")
    ):
        self.test_pdfs_dir = test_pdfs_dir
        self.reference_dir = reference_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = FailureAnalyzer()
        self.prompt_optimizer = PromptOptimizer()
        self.checkpoints = []
    
    def run_iteration(
        self,
        iteration: int,
        config: Any,  # PipelineConfig
        evaluator: Any  # SystemEvaluator
    ) -> Dict[str, Any]:
        """Run one training iteration."""
        logger.info(f"Training iteration {iteration}")
        
        # 1. Evaluate current system
        pdf_files = list(self.test_pdfs_dir.glob("*.pdf"))
        results = []
        
        for pdf_path in pdf_files:
            reference_path = None
            if self.reference_dir:
                reference_path = self.reference_dir / f"{pdf_path.stem}.json"
                if not reference_path.exists():
                    reference_path = None
            
            result = evaluator.evaluate_pdf(
                pdf_path=pdf_path,
                reference_path=reference_path,
                output_dir=self.output_dir / f"iter_{iteration}"
            )
            results.append(result)
        
        # 2. Analyze failures
        # Note: Need document objects for analysis - would need to re-run or cache
        # For now, analyze from results
        training_examples = []
        for result in results:
            if result.get('success'):
                # Extract examples from metrics
                # This is simplified - full implementation would need document objects
                pass
        
        # 3. Identify patterns
        patterns = self.analyzer.identify_patterns()
        
        # 4. Generate improvements
        improvements = self._generate_improvements(patterns)
        
        # 5. Save checkpoint
        checkpoint = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "patterns": [asdict(p) for p in patterns],
            "improvements": improvements,
            "aggregate_metrics": self._aggregate_results(results)
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_{iteration}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        self.checkpoints.append(checkpoint)
        
        return checkpoint
    
    def _generate_improvements(self, patterns: List[FailurePattern]) -> Dict[str, Any]:
        """Generate improvement suggestions."""
        improvements = {
            "prompt_changes": [],
            "masking_rule_changes": [],
            "config_changes": []
        }
        
        for pattern in patterns:
            if "masking" in pattern.pattern_type:
                improvements["masking_rule_changes"].append({
                    "pattern": pattern.description,
                    "suggestion": pattern.suggested_fix
                })
            elif pattern.pattern_type == "quality":
                improvements["prompt_changes"].append({
                    "pattern": pattern.description,
                    "suggestion": pattern.suggested_fix
                })
            elif pattern.pattern_type == "coverage":
                improvements["config_changes"].append({
                    "pattern": pattern.description,
                    "suggestion": "Increase retry count or enable fallback backend"
                })
        
        return improvements
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across PDFs."""
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            return {}
        
        return {
            "success_rate": len(successful) / len(results),
            "average_coverage": sum(r.get('coverage', 0) for r in successful) / len(successful),
            "average_bleu": sum(r.get('bleu', 0) or 0 for r in successful) / len(successful),
            "average_chrf": sum(r.get('chrf', 0) or 0 for r in successful) / len(successful),
        }


def main():
    """Example usage."""
    import sys
    from pathlib import Path
    
    test_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_data/source_pdfs")
    ref_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    loop = TrainingLoop(
        test_pdfs_dir=test_dir,
        reference_dir=ref_dir
    )
    
    # Run training iteration
    from scitran.core.pipeline import PipelineConfig
    from scripts.evaluate_system import SystemEvaluator
    
    config = PipelineConfig()
    evaluator = SystemEvaluator(config)
    
    checkpoint = loop.run_iteration(1, config, evaluator)
    print(f"Training iteration complete. Checkpoint saved.")


if __name__ == "__main__":
    main()


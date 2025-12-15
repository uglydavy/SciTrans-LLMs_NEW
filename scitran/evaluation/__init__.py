"""
Evaluation metrics for translation quality assessment.

This module provides research-grade metrics for thesis evaluation:
- BLEU score
- chrF score
- COMET score (optional)
- Glossary adherence
- Numeric consistency
- Layout fidelity proxies
"""

from scitran.evaluation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_comet,
    compute_glossary_adherence,
    compute_numeric_consistency,
    compute_layout_fidelity,
    evaluate_translation
)
from scitran.evaluation.baselines import (
    run_external_baseline,
    load_baseline_results,
)

__all__ = [
    'compute_bleu',
    'compute_chrf',
    'compute_comet',
    'compute_glossary_adherence',
    'compute_numeric_consistency',
    'compute_layout_fidelity',
    'evaluate_translation',
    'run_external_baseline',
    'load_baseline_results',
]


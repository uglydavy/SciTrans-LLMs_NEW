"""
Training module for SciTrans system improvement.

This module provides:
- Failure analysis
- Pattern identification
- Prompt optimization
- Training loop
"""

from scitran.training.trainer import (
    TrainingExample,
    FailurePattern,
    FailureAnalyzer,
    PromptOptimizer,
    TrainingLoop
)

__all__ = [
    'TrainingExample',
    'FailurePattern',
    'FailureAnalyzer',
    'PromptOptimizer',
    'TrainingLoop'
]


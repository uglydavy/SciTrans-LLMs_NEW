"""
SciTrans-LLMs NEW: Advanced Scientific Document Translation

A state-of-the-art scientific document translation system implementing
three key innovations:

1. Terminology-Constrained Translation with Advanced Masking
2. Document-Level Context with Multi-Candidate Reranking
3. Complete Layout Preservation with YOLO Detection

Usage:
    from scitran import TranslationPipeline, PipelineConfig, Document
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="cascade"
    )
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
"""

__version__ = "2.0.0"
__author__ = "SciTrans Team"
__license__ = "MIT"

# Core exports
from scitran.core.models import (
    Document,
    Block,
    Segment,
    BoundingBox,
    FontInfo,
    MaskInfo,
    BlockType,
    TranslationResult,
    TranslationMetadata
)

from scitran.core.pipeline import (
    TranslationPipeline,
    PipelineConfig
)

# Masking
from scitran.masking.engine import (
    MaskingEngine,
    MaskingConfig
)

# Scoring
from scitran.scoring.reranker import (
    AdvancedReranker,
    ScoringStrategy,
    MultiDimensionalScorer
)

# Extraction
from scitran.extraction.pdf_parser import PDFParser

# Rendering
from scitran.rendering.pdf_renderer import PDFRenderer

# Translation
from scitran.translation.base import (
    TranslationBackend,
    TranslationRequest,
    TranslationResponse
)

from scitran.translation.prompts import (
    PromptOptimizer,
    PromptLibrary,
    PromptTemplate,
    PromptStrategy
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Core
    "Document",
    "Block",
    "Segment",
    "BoundingBox",
    "FontInfo",
    "MaskInfo",
    "BlockType",
    "TranslationResult",
    "TranslationMetadata",
    "TranslationPipeline",
    "PipelineConfig",
    
    # Masking
    "MaskingEngine",
    "MaskingConfig",
    
    # Scoring
    "AdvancedReranker",
    "ScoringStrategy",
    "MultiDimensionalScorer",
    
    # Extraction
    "PDFParser",
    
    # Rendering
    "PDFRenderer",
    
    # Translation
    "TranslationBackend",
    "TranslationRequest",
    "TranslationResponse",
    "PromptOptimizer",
    "PromptLibrary",
    "PromptTemplate",
    "PromptStrategy",
]


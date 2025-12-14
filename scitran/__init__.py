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

# Build __all__ dynamically based on what imports successfully
__all__ = [
    "__version__",
    "__author__",
    "__license__",
]

# Core models - always available
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
__all__.extend([
    "Document", "Block", "Segment", "BoundingBox", "FontInfo",
    "MaskInfo", "BlockType", "TranslationResult", "TranslationMetadata"
])

# Pipeline - requires scoring module
try:
    from scitran.core.pipeline import (
        TranslationPipeline,
        PipelineConfig
    )
    __all__.extend(["TranslationPipeline", "PipelineConfig"])
except ImportError as e:
    import logging
    logging.warning(f"Pipeline not available: {e}")

# Masking - always available
from scitran.masking.engine import (
    MaskingEngine,
    MaskingConfig
)
__all__.extend(["MaskingEngine", "MaskingConfig"])

# Scoring - may require numpy
try:
    from scitran.scoring.reranker import (
        AdvancedReranker,
        ScoringStrategy,
        MultiDimensionalScorer
    )
    __all__.extend(["AdvancedReranker", "ScoringStrategy", "MultiDimensionalScorer"])
except ImportError as e:
    import logging
    logging.warning(f"Reranker not available: {e}")

# Extraction - requires PyMuPDF
try:
    from scitran.extraction.pdf_parser import PDFParser
    __all__.append("PDFParser")
except ImportError:
    pass

# Rendering - requires PyMuPDF
try:
    from scitran.rendering.pdf_renderer import PDFRenderer
    __all__.append("PDFRenderer")
except ImportError:
    pass

# Translation base - always available
from scitran.translation.base import (
    TranslationBackend,
    TranslationRequest,
    TranslationResponse
)
__all__.extend(["TranslationBackend", "TranslationRequest", "TranslationResponse"])

# Prompts - always available
from scitran.translation.prompts import (
    PromptOptimizer,
    PromptLibrary,
    PromptTemplate,
    PromptStrategy
)
__all__.extend(["PromptOptimizer", "PromptLibrary", "PromptTemplate", "PromptStrategy"])


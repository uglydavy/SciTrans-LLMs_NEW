"""
Evaluation metrics for translation quality assessment.

This module provides lightweight, deterministic helpers for:
- BLEU (sacrebleu)
- chrF (sacrebleu)
- COMET (optional dependency; returns None when unavailable)
- Glossary adherence
- Numeric consistency
- Layout fidelity proxy
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _safe_import_comet():
    """Import COMET lazily with a clear error message if missing."""
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore

        return download_model, load_from_checkpoint
    except Exception:  # pragma: no cover - purely defensive
        return None, None


def compute_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> Optional[float]:
    """Compute corpus BLEU (0-1). Returns None when inputs are empty."""
    if not hypotheses or not references:
        return None
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length.")

    try:
        import sacrebleu
    except ImportError:  # pragma: no cover - dependency guard
        logger.warning("sacrebleu not installed; BLEU unavailable.")
        return None

    score = sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score / 100.0
    return score


def compute_chrf(hypotheses: Sequence[str], references: Sequence[str]) -> Optional[float]:
    """Compute corpus chrF (0-1). Returns None when inputs are empty."""
    if not hypotheses or not references:
        return None
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length.")

    try:
        import sacrebleu
    except ImportError:  # pragma: no cover - dependency guard
        logger.warning("sacrebleu not installed; chrF unavailable.")
        return None

    score = sacrebleu.corpus_chrf(list(hypotheses), [list(references)]).score / 100.0
    return score


def compute_comet(
    hypotheses: Sequence[str],
    references: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    model_name: str = "Unbabel/wmt22-comet-da",
) -> Optional[float]:
    """Compute COMET score (optional). Returns None if COMET is unavailable or inputs are empty."""
    if not hypotheses or not references:
        return None
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length.")

    download_model, load_from_checkpoint = _safe_import_comet()
    if not download_model or not load_from_checkpoint:
        logger.info("COMET not available; returning None.")
        return None

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = []
    for idx, (hyp, ref) in enumerate(zip(hypotheses, references)):
        sample: Dict[str, Any] = {"mt": hyp, "ref": ref}
        if sources:
            sample["src"] = sources[idx]
        data.append(sample)

    scores = model.predict(data, batch_size=4, gpus=0, progress_bar=False)
    # Some versions return namedtuple, others dict
    if isinstance(scores, (list, tuple)) and scores:
        val = scores[0] if isinstance(scores[0], (float, int)) else getattr(scores[0], "score", None)
    else:
        val = getattr(scores, "system_score", None)
    return float(val) if val is not None else None


def compute_glossary_adherence(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    glossary: Dict[str, str],
) -> Dict[str, Any]:
    """Compute glossary adherence: enforced / found (1.0 if nothing to enforce)."""
    if not sources or not hypotheses or not glossary:
        return {"adherence": 1.0, "terms_found": 0, "terms_enforced": 0}
    if len(sources) != len(hypotheses):
        raise ValueError("Sources and hypotheses must have the same length.")

    def _ci_contains(text: str, needle: str) -> bool:
        return needle.lower() in text.lower()

    terms_found = 0
    terms_enforced = 0
    for src, hyp in zip(sources, hypotheses):
        for term, target in glossary.items():
            if not term or not target:
                continue
            if _ci_contains(src, term):
                terms_found += 1
                if _ci_contains(hyp, target):
                    terms_enforced += 1

    adherence = (terms_enforced / terms_found) if terms_found > 0 else 1.0
    return {
        "adherence": adherence,
        "terms_found": terms_found,
        "terms_enforced": terms_enforced,
    }


_NUM_PATTERN = re.compile(r"[+-]?\d+(?:[.,]\d+)?")


def _normalize_number(token: str) -> str:
    """Normalize numeric token for comparison (strip thousand separators, unify decimal)."""
    cleaned = token.replace(" ", "")
    # Convert French decimal comma to dot
    cleaned = cleaned.replace(",", ".")
    return cleaned


def compute_numeric_consistency(
    sources: Sequence[str],
    hypotheses: Sequence[str],
) -> Dict[str, Any]:
    """Check numeric consistency: fraction of source numbers preserved in hypothesis."""
    if not sources or not hypotheses:
        return {"consistency": 1.0, "expected": 0, "preserved": 0}
    if len(sources) != len(hypotheses):
        raise ValueError("Sources and hypotheses must have the same length.")

    expected = 0
    preserved = 0
    for src, hyp in zip(sources, hypotheses):
        src_nums = {_normalize_number(m.group()) for m in _NUM_PATTERN.finditer(src)}
        hyp_nums = {_normalize_number(m.group()) for m in _NUM_PATTERN.finditer(hyp)}
        expected += len(src_nums)
        preserved += len(src_nums & hyp_nums)

    consistency = (preserved / expected) if expected > 0 else 1.0
    return {"consistency": consistency, "expected": expected, "preserved": preserved}


@dataclass
class LayoutBlock:
    """Minimal block representation for layout fidelity."""

    bbox: Optional[Any] = None
    translated_text: Optional[str] = None


def compute_layout_fidelity(blocks: Optional[Iterable[LayoutBlock]]) -> Dict[str, Any]:
    """Proxy metric: fraction of blocks with layout info that received translations."""
    if not blocks:
        return {"layout_preservation": 1.0, "count": 0, "translated": 0}

    blocks_with_layout = [b for b in blocks if getattr(b, "bbox", None)]
    if not blocks_with_layout:
        return {"layout_preservation": 1.0, "count": 0, "translated": 0}

    translated = [
        b
        for b in blocks_with_layout
        if getattr(b, "translated_text", None) and str(getattr(b, "translated_text")).strip()
    ]
    ratio = len(translated) / len(blocks_with_layout)
    return {"layout_preservation": ratio, "count": len(blocks_with_layout), "translated": len(translated)}


def evaluate_translation(
    hypotheses: Sequence[str],
    references: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    glossary: Optional[Dict[str, str]] = None,
    blocks: Optional[Iterable[LayoutBlock]] = None,
    include_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
) -> Dict[str, Any]:
    """Compute a bundle of metrics for a system output."""
    metrics: Dict[str, Any] = {}

    # BLEU/chrF
    if references:
        metrics["bleu"] = compute_bleu(hypotheses, references)
        metrics["chrf"] = compute_chrf(hypotheses, references)
    else:
        metrics["bleu"] = None
        metrics["chrf"] = None

    # COMET (optional)
    metrics["comet"] = compute_comet(hypotheses, references, sources, comet_model) if (include_comet and references) else None

    # Glossary adherence
    if glossary and sources:
        metrics.update(
            {
                "glossary_adherence": compute_glossary_adherence(sources, hypotheses, glossary)["adherence"],
                "glossary_terms_found": compute_glossary_adherence(sources, hypotheses, glossary)["terms_found"],
                "glossary_terms_enforced": compute_glossary_adherence(sources, hypotheses, glossary)["terms_enforced"],
            }
        )
    else:
        metrics["glossary_adherence"] = None
        metrics["glossary_terms_found"] = 0
        metrics["glossary_terms_enforced"] = 0

    # Numeric consistency
    metrics.update(compute_numeric_consistency(sources or [], hypotheses))

    # Layout fidelity
    layout_stats = compute_layout_fidelity(blocks)
    metrics["layout_preservation"] = layout_stats["layout_preservation"]
    metrics["layout_blocks_with_bbox"] = layout_stats["count"]
    metrics["layout_blocks_translated"] = layout_stats["translated"]

    return metrics



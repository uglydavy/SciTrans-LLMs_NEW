import math
import pytest

from scitran.evaluation.metrics import (
    LayoutBlock,
    compute_bleu,
    compute_chrf,
    compute_comet,
    compute_glossary_adherence,
    compute_numeric_consistency,
    compute_layout_fidelity,
    evaluate_translation,
)


def test_bleu_perfect_match():
    pytest.importorskip("sacrebleu")
    refs = ["The cat sits on the mat.", "Numbers: 3.14"]
    hyps = ["The cat sits on the mat.", "Numbers: 3.14"]
    score = compute_bleu(hyps, refs)
    assert score is not None
    assert math.isclose(score, 1.0, rel_tol=1e-6)


def test_chrf_perfect_match():
    pytest.importorskip("sacrebleu")
    refs = ["hello world", "quick brown fox"]
    hyps = ["hello world", "quick brown fox"]
    score = compute_chrf(hyps, refs)
    assert score is not None
    assert math.isclose(score, 1.0, rel_tol=1e-6)


def test_glossary_adherence_counts():
    sources = ["The neuron fires", "A synapse connects neurons"]
    hyps = ["Le neurone tire", "Une synapse relie les neurones"]
    glossary = {"neuron": "neurone"}
    stats = compute_glossary_adherence(sources, hyps, glossary)
    assert stats["terms_found"] == 2
    assert stats["terms_enforced"] == 2
    assert math.isclose(stats["adherence"], 1.0, rel_tol=1e-6)


def test_numeric_consistency_full_and_missing():
    sources = ["Value 3.14 and 42", "No numbers here"]
    hyps = ["Valeur 3,14 et 42", "Pas de nombres ici"]
    stats = compute_numeric_consistency(sources, hyps)
    assert stats["expected"] == 2
    assert stats["preserved"] == 2
    assert math.isclose(stats["consistency"], 1.0, rel_tol=1e-6)

    hyps_missing = ["Valeur 3,14", "Pas de nombres ici"]
    stats_missing = compute_numeric_consistency(sources, hyps_missing)
    assert stats_missing["expected"] == 2
    assert stats_missing["preserved"] == 1
    assert math.isclose(stats_missing["consistency"], 0.5, rel_tol=1e-6)


def test_layout_fidelity_ratio():
    blocks = [
        LayoutBlock(bbox=(0, 0, 10, 10), translated_text="ok"),
        LayoutBlock(bbox=(0, 0, 10, 10), translated_text=None),
        LayoutBlock(bbox=None, translated_text="ignored"),  # no layout info
    ]
    stats = compute_layout_fidelity(blocks)
    assert stats["count"] == 2
    assert stats["translated"] == 1
    assert math.isclose(stats["layout_preservation"], 0.5, rel_tol=1e-6)


def test_evaluate_translation_bundle():
    pytest.importorskip("sacrebleu")
    refs = ["Hello world", "The number is 42"]
    hyps = ["Hello world", "The number is 42"]  # identical to make BLEU/chrF = 1
    sources = ["Hello world", "The number is 42"]
    glossary = {"number": "number"}  # aligned for perfect adherence
    blocks = [LayoutBlock(bbox=(0, 0, 1, 1), translated_text="Hello world")]

    metrics = evaluate_translation(
        hypotheses=hyps,
        references=refs,
        sources=sources,
        glossary=glossary,
        blocks=blocks,
        include_comet=False,
    )

    assert math.isclose(metrics["bleu"], 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["chrf"], 1.0, rel_tol=1e-6)
    assert metrics["comet"] is None  # not requested
    assert metrics["glossary_terms_found"] == 1
    assert metrics["glossary_terms_enforced"] == 1
    assert math.isclose(metrics["glossary_adherence"] or 0.0, 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["consistency"], 1.0, rel_tol=1e-6)
    assert metrics["layout_blocks_with_bbox"] == 1
    assert metrics["layout_blocks_translated"] == 1
    assert math.isclose(metrics["layout_preservation"], 1.0, rel_tol=1e-6)


def test_compute_comet_returns_none_when_missing():
    # When COMET is not installed, function should return None, not raise
    hyps = ["hello"]
    refs = ["bonjour"]
    score = compute_comet(hypotheses=hyps, references=refs, sources=None, model_name="Unbabel/wmt22-comet-da")
    assert score is None



# SPRINT 5 — Evaluation Harness (Complete)

## Scope & goals
- Provide thesis-grade evaluation metrics (BLEU/chrF, glossary adherence, numeric consistency, layout proxy; COMET optional).
- Add an experiment runner CLI for reproducible scoring.
- Provide hooks for external baselines without bundling AGPL code.

## Delivered
- Metrics module: `scitran/evaluation/metrics.py`
  - BLEU/chrF via sacrebleu (gracefully returns None if sacrebleu missing).
  - COMET optional/guarded (returns None if unavailable).
  - Glossary adherence, numeric consistency (handles decimal comma), layout fidelity proxy, bundled `evaluate_translation`.
- Baseline hooks: `scitran/evaluation/baselines.py`
  - Run external baselines safely; load JSON results.
- Experiment runner: `scripts/run_experiment.py`
  - CLI: hyp/ref/src + glossary; optional COMET; JSON output.
- Tests: `tests/unit/test_evaluation_metrics.py`
  - Glossary adherence, numeric consistency, layout proxy, evaluation bundle, COMET-missing path; BLEU/chrF now run with sacrebleu installed.

## Notes
- COMET remains optional (environment-dependent). Code paths are guarded.
- sacrebleu installed to remove skips for BLEU/chrF tests.

## Tests
- Unit: 70/70 passing (offline, deterministic; benign PyMuPDF deprecation warnings).***


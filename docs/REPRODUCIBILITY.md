# Reproducibility Guide

Authoritative steps to reproduce results and run deterministic tests.

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[all]
# Optional metrics
pip install sacrebleu          # BLEU/chrF
# COMET (optional; heavy): pip install unbabel-comet
```

Pin environment by exporting `pip freeze > requirements-lock.txt` if needed.

## Tests
Deterministic, offline:
```bash
pytest tests/unit/ -v
```
Current status: 70 passed, 0 skipped (with sacrebleu installed).

## Configs
- Single source of truth: `PipelineConfig` (CLI/GUI use same fields).
- Key reproducibility flags:
  - `strict_mode=true`
  - `max_translation_retries`, `retry_backoff_factor`
  - `enable_fallback_backend`, `fallback_backend`
  - `enable_glossary`, `glossary_domains`, `glossary_path`
  - `enable_context`, `enable_refinement`, `enable_adaptive_escalation`

## Evaluation
```bash
python scripts/run_experiment.py \
  --hyp outputs.txt --ref references.txt --src sources.txt \
  --glossary glossary.json --include-comet \
  --out results.json
```
Outputs include BLEU/chrF (if sacrebleu installed), glossary adherence, numeric consistency, layout proxy, and COMET when available.

## Artifacts
- Strict mode emits machine-readable failure reports (JSON).
- Keep run logs and configs alongside results for thesis tables.

## No-network requirement for tests
- Unit tests use dummy translators; no API calls.
- COMET path is guarded; safe when dependency missing.


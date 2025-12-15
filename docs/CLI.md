# CLI Guide

Authoritative commands and flags for SciTrans-LLMs.

## Basics
```bash
# GUI
./scitrans gui

# Translate PDF (strict + fallback)
./scitrans translate input.pdf --backend cascade --source en --target fr \
  --strict-mode true --max-translation-retries 2 --enable-fallback-backend true
```

## Key flags
- `--backend`: cascade | openai | anthropic | deepseek | free | huggingface | ollama
- `--source / --target`: language codes (e.g., en, fr)
- `--strict-mode`: default true; abort on missing translations and emit JSON report
- `--max-translation-retries`: retries before fallback
- `--retry-backoff-factor`: exponential backoff factor (default 2.0)
- `--enable-fallback-backend`: escalate to stronger backend on failure
- `--fallback-backend`: backend name for escalation
- `--enable-glossary`: enforce glossary (on by default)
- `--glossary-path`: JSON glossary (term -> translation)
- `--glossary-domains`: comma list of bundled domains (ml, physics, biology, cs, chemistry, statistics, europarl)
- `--enable-refinement`: document-level refinement (default on)
- `--enable-context`: document-level context (default on)
- `--enable-mask ing`: disable only for ablations/baselines
- `--enable-adaptive-escalation`: control adaptive controller

## Testing
```bash
pytest tests/unit/ -v
```

## Troubleshooting
- Missing translations: ensure `--strict-mode true` and fallback configured.
- Optional metrics deps: `pip install sacrebleu` (BLEU/chrF); `pip install unbabel-comet` (COMET, heavy).


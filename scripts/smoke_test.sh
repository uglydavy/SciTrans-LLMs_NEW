#!/usr/bin/env bash
set -euo pipefail

# Simple offline smoke test: import pipeline and build a minimal config.
python - <<'PY'
from scitran.core.pipeline import TranslationPipeline, PipelineConfig

config = PipelineConfig(
    source_lang="en",
    target_lang="fr",
    backend="cascade",
    strict_mode=True,
    max_translation_retries=1,
    enable_fallback_backend=False,
    enable_glossary=False,
    enable_context=False,
    enable_refinement=False,
    enable_adaptive_escalation=False,
)

pipeline = TranslationPipeline(config)
print("Smoke test OK: pipeline constructed.")
PY


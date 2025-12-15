# Changelog

## v1.0.0
- Translation coverage guarantee: retries, fallback backend, strict mode failure reports.
- Glossary enforcement via `GlossaryManager`; document-level context and refinement with constraint safety.
- Evaluation harness: BLEU/chrF, glossary adherence, numeric consistency, layout proxy; COMET optional.
- Ablation flags for masking/glossary/context/refinement/adaptive escalation.
- Tests: 70/70 unit tests passing (offline, deterministic).
- Docs consolidated (CLI, GUI, Evaluation, Ablations, Reproducibility, Deployment).
- CI/CD: GitHub Actions for smoke + unit tests; tag builds/pushes Docker image to GHCR.
- Packaging: sample configs, env example, smoke test, Dockerfile refresh.


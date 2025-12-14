# SciTrans-LLMs NEW ![CI](https://github.com/uglydavy/scitrans-llms_NEW/actions/workflows/ci.yml/badge.svg)

Layout-preserving EN↔FR translation with glossary enforcement, document-level context/refinement, and adaptive escalation. Strict coverage guarantees ensure no silent partial translations.

## Highlights
- Coverage guarantee: retries, fallback backend, strict mode JSON failure reports.
- Masking for LaTeX/code/URLs; glossary enforcement via `GlossaryManager`.
- Document-level context + refinement with constraint safety (placeholders + glossary).
- Ablation flags (masking, glossary, context, refinement, adaptive escalation).
- Evaluation harness: BLEU/chrF, glossary adherence, numeric consistency, layout proxy; COMET optional.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[all]
# Optional eval extras
pip install sacrebleu          # BLEU/chrF
# COMET (optional; heavy): pip install unbabel-comet
```

## Quick start
```bash
# GUI
./scitrans gui

# CLI translation (strict + fallback)
./scitrans translate input.pdf --backend cascade --source en --target fr \
  --strict-mode true --max-translation-retries 2 --enable-fallback-backend true

# Tests (deterministic, offline)
pytest tests/unit/ -v
```

## Evaluation
```bash
python scripts/run_experiment.py \
  --hyp outputs.txt --ref references.txt --src sources.txt \
  --glossary glossary.json --include-comet \
  --out results.json
```

## Build / deploy
```bash
# Build wheel/sdist
pip install build
python -m build

# Smoke test (offline)
./scripts/smoke_test.sh
```

## CI/CD
- GitHub Actions workflow runs smoke + unit tests on push/PR to main.
- Tags `v*` build and push Docker image to `ghcr.io/uglydavy/scitrans-llms:<tag>`.
- See `docs/CI_CD.md` for details.

## Docs (authoritative)
- `docs/ARCHITECTURE.md` – system & pipeline overview
- `docs/CLI.md` – CLI commands and flags
- `docs/GUI.md` – GUI usage
- `docs/EVALUATION.md` – metrics and experiment runner
- `docs/ABLATIONS.md` – toggleable features for studies
- `docs/REPRODUCIBILITY.md` – environment, tests, configs
- Legacy/older notes: see `docs/LEGACY.md`

## License
MIT


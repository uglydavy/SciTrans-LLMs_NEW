# SciTrans-LLMs NEW ![CI](https://github.com/uglydavy/scitrans-llms_NEW/actions/workflows/ci.yml/badge.svg)

Layout-preserving EN↔FR translation with glossary enforcement, document-level context/refinement, and adaptive escalation. Strict coverage guarantees ensure no silent partial translations.

## Highlights
- Coverage guarantee: retries, fallback backend, strict mode JSON failure reports.
- Masking for LaTeX/code/URLs; glossary enforcement via `GlossaryManager`.
- Document-level context + refinement with constraint safety (placeholders + glossary).
- Ablation flags (masking, glossary, context, refinement, adaptive escalation).
- Evaluation harness: BLEU/chrF, glossary adherence, numeric consistency, layout proxy; COMET optional.
- **NEW**: Enhanced layout-preserving pipeline with cross-page continuity handling (`improvements/` package).

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

# New improvements pipeline (cross-page continuity + layout preservation)
python -c "
from improvements import translate_pdf
result = translate_pdf(
    'input.pdf',
    'output.pdf',
    target_lang='fr',
    backend='cascade',
    glossary_domains=['ml', 'physics']
)
print(f'Layout score: {result[\"layout_score\"]:.2f}')
"

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

## New Improvements Package

The `improvements/` package provides an enhanced translation pipeline with:

1. **Layout Extraction** (`layout_parser.py`): Extracts text and non-text blocks with bounding boxes, font information, and styling from PDFs using PyMuPDF.

2. **Cross-Page Continuity** (`translation_pipeline.py`): Automatically detects when text segments continue across pages and merges them before translation, ensuring coherent sentence-level translation.

3. **Layout-Preserving Rendering** (`pdf_renderer.py`): Renders translated text back into PDFs using original fonts, sizes, and positions while preserving all non-text elements (images, drawings, etc.).

4. **Quality Metrics** (`metrics.py`): Computes layout fidelity scores (IoU on bounding boxes) and translation quality metrics (COMET, BLEU, chrF).

### Usage Example

```python
from improvements import translate_pdf, create_translator_with_glossary

# Simple usage with default translator
result = translate_pdf(
    'input.pdf',
    'output.pdf',
    target_lang='fr',
    backend='cascade',
    glossary_domains=['ml']
)

# Custom translator with glossary enforcement
translator = create_translator_with_glossary(
    backend='openai',
    source_lang='en',
    target_lang='fr',
    api_key='your-api-key',
    glossary_domains=['ml', 'physics']
)

result = translate_pdf(
    'input.pdf',
    'output.pdf',
    target_lang='fr',
    translator=translator
)
```

### Pipeline Steps

1. **Extract Layout**: Parse PDF to get block structures with bounding boxes and font info
2. **Detect Continuations**: Identify text blocks that continue across pages (end without punctuation)
3. **Merge Blocks**: Combine split blocks so translator sees complete sentences
4. **Translate**: Translate merged blocks using pluggable translator (supports all backends)
5. **Split Back**: Distribute translations back to original page/block structure
6. **Render**: Insert translated text into PDF preserving original layout
7. **Metrics**: Compute layout fidelity and translation quality scores

## Docs (authoritative)
- `docs/ARCHITECTURE.md` – system & pipeline overview
- `docs/CLI.md` – CLI commands and flags
- `docs/GUI.md` – GUI usage
- `docs/EVALUATION.md` – metrics and experiment runner
- `docs/ABLATIONS.md` – toggleable features for studies
- `docs/REPRODUCIBILITY.md` – environment, tests, configs

## License
MIT


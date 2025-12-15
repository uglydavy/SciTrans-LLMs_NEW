# Evaluation

Authoritative reference for measuring system quality.

## Metrics
- BLEU (sacrebleu)
- chrF (sacrebleu)
- COMET (optional, heavy; guarded, returns None if missing)
- Glossary adherence (terms enforced / terms found)
- Numeric consistency (source numbers preserved)
- Layout fidelity proxy (fraction of layout-aware blocks translated)

## Optional dependencies
```bash
pip install sacrebleu          # BLEU/chrF
# COMET (optional; heavy, may require older pandas/torch stacks)
# pip install unbabel-comet
```

## Experiment runner
```bash
python scripts/run_experiment.py \
  --hyp outputs.txt --ref references.txt --src sources.txt \
  --glossary glossary.json --include-comet \
  --out results.json
```

Inputs: one line per example. Glossary: JSON mapping source term -> target term.

## Interpreting outputs
- `bleu`, `chrf`: 0-1 (None if refs missing or sacrebleu not installed).
- `comet`: float or None if COMET unavailable.
- `glossary_adherence`: 0-1; plus counts found/enforced.
- `consistency`: numeric consistency ratio; `expected` and `preserved` counts.
- `layout_preservation`: ratio of layout-aware blocks translated.

## Baselines
Use external runners; hooks in `scitran/evaluation/baselines.py`:
- `run_external_baseline(command, workdir=None, timeout=3600)`
- `load_baseline_results(path)`

Avoid AGPL code; run baselines externally and import only JSON outputs.


# Release Checklist

Use this checklist before tagging/publishing a release.

## 1) Tests & smoke (offline, deterministic)
- [ ] `./scripts/ci.sh` (smoke + unit tests)
- [ ] Optional: integration/e2e if enabled

## 2) Optional eval dependencies
- [ ] `pip install sacrebleu` (BLEU/chrF)
- [ ] (Optional, environment-sensitive) `pip install unbabel-comet` — only if COMET required and Python/pandas/torch compatible

## 3) Build artifacts
- [ ] `pip install build`
- [ ] `python -m build`
- [ ] Check `dist/*.whl` and `dist/*.tar.gz`

## 4) Docker image (slim CPU)
- [ ] `docker build -t scitrans-llms .`
- [ ] Smoke run: `docker run -p 7860:7860 -v ~/.scitrans:/root/.scitrans scitrans-llms ./scitrans gui`

## 5) Config & env samples
- [ ] `configs/example.yaml` reviewed/updated
- [ ] `configs/env.example` filled locally (do not commit secrets)

## 6) Docs
- [ ] README and `docs/` (CLI, GUI, EVALUATION, ABLATIONS, REPRODUCIBILITY, DEPLOYMENT) are current

## 7) Tag and publish (run manually)
- [ ] Working tree clean
- [ ] Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] (Optional) Publish wheel/sdist (e.g., twine) and Docker image to your registry


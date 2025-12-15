# CI/CD Guide

## GitHub Actions (provided)
- Workflow: `.github/workflows/ci.yml`
- Triggers:
  - Push/PR to `main`: run smoke + unit tests (offline, deterministic) via `scripts/ci.sh`.
  - Tags `v*`: build and push Docker image to `ghcr.io/uglydavy/scitrans-llms:<tag>`.
- Requirements:
  - Uses `GITHUB_TOKEN` with `packages: write` for GHCR.
  - Builds on Python 3.11, installs `.[dev,gui]`.

## Local release steps (manual)
1) `./scripts/ci.sh`
2) `python -m build`
3) Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`; `git push origin vX.Y.Z`
4) Docker (GHCR):
   ```bash
   docker build -t scitrans-llms:latest .
   docker tag scitrans-llms:latest ghcr.io/uglydavy/scitrans-llms:vX.Y.Z
   echo $GHCR_TOKEN | docker login ghcr.io -u uglydavy --password-stdin
   docker push ghcr.io/uglydavy/scitrans-llms:vX.Y.Z
   ```
5) Optional: upload wheel/sdist with twine.

## Notes
- COMET remains optional; tests are offline and deterministic.
- For private registries, adjust `tags` and login in the workflow.


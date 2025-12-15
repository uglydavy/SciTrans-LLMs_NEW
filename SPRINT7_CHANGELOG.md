# SPRINT 7 — Release & Ops (Complete)

## Scope & goals
- Make the project release-ready with packaging, Docker, CI/CD, and release guidance.

## Delivered
- Deployment/packaging:
  - `docs/DEPLOYMENT.md`, `RELEASE_CHECKLIST.md`, `configs/example.yaml`, `configs/env.example`, `scripts/smoke_test.sh`.
- Docker:
  - Refreshed Dockerfile (python:3.11-slim, installs `.[gui]`, default `./scitrans gui`, healthcheck).
- CI/CD:
  - GitHub Actions `.github/workflows/ci.yml`: push/PR runs smoke + unit; tags `v*` build/push Docker to `ghcr.io/uglydavy/scitrans-llms:<tag>`.
  - `docs/CI_CD.md` documents workflow and manual release steps.
- Docs/metadata:
  - README adds CI badge and CI/CD section.
  - `CHANGELOG.md` with v1.0.0 entry.

## Tests
- Unit: 70/70 passing (offline; benign PyMuPDF deprecation warnings).***


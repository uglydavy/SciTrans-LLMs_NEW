#!/usr/bin/env bash
set -euo pipefail

# Minimal CI script: smoke test + unit tests (offline, deterministic)
./scripts/smoke_test.sh
pytest tests/unit/ -q


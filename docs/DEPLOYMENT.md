# Deployment & Release

Authoritative guide for packaging, Docker, and runtime configuration.

## Packaging (wheel/sdist)
```bash
# In venv
pip install build
python -m build
# Artifacts: dist/*.whl and dist/*.tar.gz
```

Smoke test + unit tests:
```bash
./scripts/ci.sh
```

## Docker (slim CPU)
Example Docker build/run:
```bash
docker build -t scitrans-llms .
docker run -p 7860:7860 -v ~/.scitrans:/root/.scitrans scitrans-llms ./scitrans gui
```

## Docker Compose (GUI + optional cache)
```yaml
services:
  scitrans:
    image: scitrans-llms:latest
    ports: ["7860:7860"]
    env_file: .env
    volumes:
      - ~/.scitrans:/root/.scitrans
    command: ["./scitrans", "gui"]
  # redis:
  #   image: redis:7
  #   ports: ["6379:6379"]
```

## Runtime config
- CLI/GUI share the same `PipelineConfig` fields.
- Example config: `configs/example.yaml` (strict mode, retries, fallback, glossary, refinement, ablations).
- Environment: `.env.example` shows typical keys; never commit real keys.

## Smoke test (offline)
```bash
./scripts/smoke_test.sh
```
Validates imports and basic pipeline construction without network calls.

## Notes
- COMET is optional; install only if needed (`pip install unbabel-comet`) and consider Python version compatibility.
- Use strict mode for production/thesis runs to avoid partial translations.


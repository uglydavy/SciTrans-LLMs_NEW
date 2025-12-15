FROM python:3.11-slim

WORKDIR /app

# System deps (slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata first (better cache)
COPY pyproject.toml ./
COPY requirements.txt requirements-core.txt requirements-ml.txt requirements-dev.txt requirements-minimal.txt ./

# Install project (GUI extras, no dev/ML by default)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[gui]"

# Copy source
COPY scitran/ ./scitran/
COPY cli/ ./cli/
COPY gui/ ./gui/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY scitrans ./scitrans
COPY README.md .

# Runtime env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCITRANS_CACHE_DIR=/app/cache \
    SCITRANS_OUTPUT_DIR=/app/output

RUN mkdir -p /app/output /app/cache /app/data

EXPOSE 7860

# Health check: import pipeline
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from scitran.core.pipeline import TranslationPipeline; print('OK')" || exit 1

# Default: launch GUI
# Note: scitrans script is copied but in Docker we use Python directly
# The script is available for manual use via docker exec
RUN chmod +x ./scitrans
# Use Python directly in Docker (no venv needed)
CMD ["python", "-m", "cli.commands.main", "gui"]


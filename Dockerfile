# ==============================================================================
# Self-Healing MLOps Runtime - Serving Container
# FastAPI + ONNX Runtime + Confluent Kafka Telemetry Layer
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set security and Python runtime flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml /app/

# Install Python production dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy source code and default artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY config/ /app/config/

# Ensure appropriate permissions for application user
RUN chown -R appuser:appuser /app

# Switch to unprivileged user
USER appuser

EXPOSE 8000

# Container healthcheck querying FastAPI /health endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch low-latency uvicorn ASGI server
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


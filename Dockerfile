# Phase 4E — ML service image (multi-stage, slim base).
#
# Stage 1 (builder) installs Python deps into a venv; stage 2 copies just
# the venv + source, so the final image carries no build toolchain.

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# build tools needed to compile a few wheels (e.g. confluent-kafka).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ML_SERVICE_HOST=0.0.0.0 \
    ML_SERVICE_PORT=8000

# libgomp1 is required at runtime by lightgbm / xgboost (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness baked into the image so orchestrators get health for free.
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "services.ml_service.server"]

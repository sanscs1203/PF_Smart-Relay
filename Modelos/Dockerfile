# =============================================================================
# Dockerfile — Smart Relay Project
# Pipeline: preprocess → split → train → evaluate → mcdm (IEEE 5-bus)
#
# Build:
#   docker build -t smart-relay .
#
# Run (via docker-compose — recomendado):
#   docker compose up
# =============================================================================

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project source ────────────────────────────────────────────────────────────
# Solo se copian los archivos de código — data/ se monta como volumen
COPY utils/   utils/
COPY ieee5/   ieee5/
COPY ieee13/  ieee13/

# ── Reproducibilidad ──────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# Dockerfile — Smart Relay Project (Jetson Nano / JetPack R32.7.2)
# =============================================================================
# Build:
#   docker build -t smart-relay .
#
# Run (via docker-compose):
#   docker compose up
#
# Compatible with:
#   - NVIDIA Jetson Nano (L4T R32.7.2)
#   - JetPack 4.6.2 (CUDA 10.2 / cuDNN 8.0)
# =============================================================================

# ── Base image (L4T with CUDA & cuDNN) ──────────────────────────────────────
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# ── Install Python 3.8 and system dependencies ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        build-essential \
        gcc \
        g++ \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        pkg-config \
        libssl-dev \
        libfreetype6-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --set python3 /usr/bin/python3.8 \
    && rm -f /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python

# ── Python dependencies ────────────────────────────────────────────────────
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Install TensorFlow for Jetson (specific wheel) ─────────────────────────
# TensorFlow version must match JetPack 4.6.x (R32.7.x) – using 2.5.0+nv21.07
RUN pip3 install --no-cache-dir --extra-index-url \
        https://developer.download.nvidia.com/compute/redist/jp/v44 \
        tensorflow==2.5.0+nv21.07

# ── Project source ─────────────────────────────────────────────────────────
# (data/ will be mounted as a volume, so we only COPY code)
COPY utils/   utils/
COPY ieee5/   ieee5/
COPY ieee13/  ieee13/

# ── Reproducibility & Jetson environment ──────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Preload libraries (optional, avoids some runtime warnings)
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Ensure numpy is imported before TF to avoid HDF5 mismatch issues
ENV OPENBLAS_CORETYPE=ARMV8
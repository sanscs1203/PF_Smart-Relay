# =============================================================================
# Dockerfile — Smart Relay Project (Jetson Nano / JetPack 4.6.x)
# =============================================================================
# This image uses the official NVIDIA TensorFlow container, which already
# includes Python 3.6, CUDA 10.2, cuDNN 8.2 and TensorFlow 2.7.
# No need to install TensorFlow or h5py manually.
# =============================================================================

FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

# ── Avoid interactive prompts ─────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ── Set locale to UTF-8 ──────────────────────────────────────────────────
# This prevents UnicodeDecodeError when reading requirements.txt
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── Install system dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Upgrade pip and build tools ───────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── Install project dependencies ──────────────────────────────────────────
# Make sure your requirements.txt does NOT include tensorflow, h5py, etc.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────
COPY utils/   utils/
COPY ieee5/   ieee5/
COPY ieee13/  ieee13/

# ── Jetson runtime optimizations ──────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Important to prevent memory allocation errors on Jetson
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# ── Default command ───────────────────────────────────────────────────────
CMD ["python3", "-u", "ieee13/main.py"]

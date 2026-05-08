# =============================================================================
# Dockerfile — Smart Relay Project (Jetson Nano / JetPack R32.7.2)
# =============================================================================
# Build:
#   docker build -t smart-relay .
#
# Run:
#   docker compose up
# =============================================================================

# ── Base image (L4T with CUDA & cuDNN) ──────────────────────────────────────
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# ── Environment variables to avoid interaction during installation ──────────
ENV DEBIAN_FRONTEND=noninteractive

# ── Install Python 3.8 and system dependencies ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        build-essential \
        gcc \
        g++ \
        cmake \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        pkg-config \
        libssl-dev \
        libfreetype6-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --set python3 /usr/bin/python3.8 \
    && rm -f /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python

# ── Python setup and h5py compilation ───────────────────────────────────────
WORKDIR /app

# Upgrade build tools (CRITICAL for aarch64 compatibility)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel Cython

# Define paths so h5py can find HDF5 serial libraries on Jetson
ENV HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial
ENV CPATH=/usr/include/hdf5/serial

# Install h5py explicitly (this version is required for TF 2.5)
# Note: This may take several minutes as it compiles from source on the Nano.
RUN pip3 install --no-cache-dir h5py==2.10.0

# ── Install TensorFlow for Jetson ──────────────────────────────────────────
# We install TF first to handle its base dependencies correctly
RUN pip3 install --no-cache-dir --extra-index-url \
        https://developer.download.nvidia.com/compute/redist/jp/v46 \
        tensorflow==2.5.0+nv21.07

# ── Project dependencies ───────────────────────────────────────────────────
# Copy requirements.txt (Ensure h5py and tensorflow are REMOVED from this file)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Project source ─────────────────────────────────────────────────────────
# (Data directory should be mounted as a volume via docker-compose)
COPY utils/   utils/
COPY ieee5/   ieee5/
COPY ieee13/  ieee13/

# ── Runtime environment and optimizations ──────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Preload libgomp to prevent memory allocation errors on Jetson
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Optimization for OpenBLAS on ARMV8 architecture
ENV OPENBLAS_CORETYPE=ARMV8

# Entry point (Adjust main.py path as needed)
CMD ["python3", "-u", "ieee13/main.py"]
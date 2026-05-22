FROM python:3.6-slim

# Variables de entorno para evitar problemas
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Solo las dependencias de sistema que sean estrictamente necesarias
# (probablemente ninguna si usas ruedas precompiladas para x86)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt que ahora SÍ debe contener tensorflow, h5py, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY utils/   utils/
COPY ieee5/   ieee5/
COPY ieee13/  ieee13/

CMD ["python3", "-u", "ieee13/main.py"]
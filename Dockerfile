FROM python:3.11-slim-bookworm

# Instalar dependencias de compilación mínimas (grpcio puede necesitar build-essential si no hay wheel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
# Instala dependencias
RUN pip install -r requirements.txt

ADD model model
ADD service service
ADD repository repository
ADD lib lib
COPY app.py app.py

EXPOSE 8080

# Docker healthcheck probing GET /health; curl -f falla si HTTP>=400 (503 cuando no listo)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health > /dev/null || exit 1

# Start with Gunicorn in production
CMD [ "gunicorn", "-b", "0.0.0.0:8080", "app:app" ]

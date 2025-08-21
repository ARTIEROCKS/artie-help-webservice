FROM python:3.11-slim-bookworm

# Instalar dependencias de compilación mínimas (grpcio puede necesitar build-essential si no hay wheel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
# Instala dependencias
RUN pip install -r requirements.txt

ADD model model
ADD service service
ADD repository repository
COPY app.py app.py

EXPOSE 8080

CMD [ "python", "app.py", "--host=0.0.0.0", "--port=8080" ]

# Help Webservice

Servicio Flask para inferencia del modelo de ayuda (help_model.h5/.keras) con endpoints de predicción y health para probes, listo para ejecutarse en Python 3.11 y Docker.

## Características
- Python 3.11, Flask 3.
- TensorFlow/Keras 2.15 con capas personalizadas registradas en `lib/keras_custom_layers.py`.
- Endpoints:
  - `GET /health` — OK cuando el modelo `model/help_model.h5` está cargado correctamente al arrancar.
  - `POST /api/v1/help-model/predict` — recibe interacciones del estudiante y devuelve si se debe mostrar ayuda.
- Dockerfile con healthcheck nativo (curl -> `/health`).

## Requisitos
- Archivo de modelo en `model/help_model.h5` (incluido en el repo junto a variantes `.keras`).
- Python 3.11 (para ejecución local) o Docker.
- Para el endpoint de predicción: acceso a MongoDB y variables de entorno configuradas.

### Variables de entorno (MongoDB)
El servicio de predicción usa MongoDB a través de `repository/db.py` y requiere:
- `APP_MONGO_HOST`
- `APP_MONGO_PORT`
- `APP_MONGO_USER`
- `APP_MONGO_PASS`
- `APP_MONGO_DB`

Ejemplo (bash):
```bash
export APP_MONGO_HOST=localhost
export APP_MONGO_PORT=27017
export APP_MONGO_USER=myuser
export APP_MONGO_PASS=mypass
export APP_MONGO_DB=artie
```

## Arranque rápido

### Opción A: Docker (recomendada)
1) Construir imagen
```bash
docker build -t help-webservice .
```
2) Ejecutar el contenedor (con probe de health incorporado)
- Solo health (sin dependencia de Mongo):
```bash
docker run -d --name help-webservice -p 8080:8080 help-webservice
```
- Con predicción (requiere Mongo y variables):
```bash
docker run -d --name help-webservice \
  -p 8080:8080 \
  -e APP_MONGO_HOST=host.docker.internal \
  -e APP_MONGO_PORT=27017 \
  -e APP_MONGO_USER=myuser \
  -e APP_MONGO_PASS=mypass \
  -e APP_MONGO_DB=artie \
  help-webservice
```
3) Probar health
```bash
curl -i http://localhost:8080/health
```
Debería responder `200 {"status":"ok"}` cuando el modelo esté cargado.

Estado de salud del contenedor (Docker):
```bash
docker ps --filter name=help-webservice --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

### Opción B: Local (venv)
1) Crear y activar entorno
```bash
python -m venv venv
source venv/bin/activate
```
2) Instalar dependencias
```bash
pip install -r requirements.txt
```
Notas:
- En macOS, se instalará `tensorflow-macos` (y `tensorflow-metal` en Apple Silicon) automáticamente por los marcadores en `requirements.txt`.
- En Linux se instalará `tensorflow==2.15.0`.

3) Lanzar el servicio
```bash
python app.py
```
4) Probar health
```bash
curl -i http://localhost:8080/health
```

## Uso del endpoint de predicción
- URL: `POST /api/v1/help-model/predict`
- Body (application/json): lista de interacciones. Campos mínimos esperados por el preprocesado: `student.id` (o `student._id`), `exercise.id` (o `exercise._id`), `lastLogin`, `dateTime`.

Ejemplo:
```bash
curl -s -X POST http://localhost:8080/api/v1/help-model/predict \
  -H 'Content-Type: application/json' \
  -d '[{
    "student": {"id": "S1"},
    "exercise": {"id": "E1"},
    "lastLogin": "2024-03-01T10:00:00.000000",
    "dateTime":  "2024-03-01T10:00:05.000000",
    "requestHelp": false
  }]'
```
Respuesta (string JSON con envoltorio `body`):
```
{"body": {"message": "OK", "object": 0}}
```
`object` es 1 si se debe mostrar ayuda; 0 en caso contrario.

## Estructura del proyecto
```
app.py                     # Flask app, endpoints /health y /predict
service/
  preprocess.py            # Transformaciones de datos
  model.py                 # Carga del modelo y predicción
  queue_service.py         # Obtención/actualización de interacciones en Mongo
repository/
  db.py                    # Cliente y operaciones Mongo (lee variables de entorno)
lib/
  keras_custom_layers.py   # Capas/funciones Keras personalizadas registradas
model/
  help_model.h5            # Modelo principal usado por defecto
  selectedfeatures.csv     # CSV con columnas esperadas por el modelo
```

## Notas de compatibilidad
- Python 3.11.
- TensorFlow/Keras 2.15.
- `requirements.txt` fija versiones que evitan compilaciones nativas problemáticas en Py3.11 (p. ej. `grpcio`, `PyYAML`, `wrapt`).
- En `app.py` se importan las funciones de `lib/keras_custom_layers.py` para que la deserialización del modelo con objetos personalizados funcione sin `custom_objects` explícito.

## Troubleshooting
- `ImportError: cannot import name 'formatargspec' from 'inspect'` — usar `wrapt==1.14.1` (ya fijado en requirements) en lugar de versiones antiguas.
- Fallo compilando `grpcio` o `PyYAML` — se usan versiones con wheels para Py3.11 (`grpcio==1.62.2`, `PyYAML==6.0.2`).
- `/health` devuelve 503 — asegurar que `model/help_model.h5` existe y es legible; revisar logs del contenedor o consola.
- Predicción falla por conexión a DB — exportar variables de entorno de Mongo y que la instancia sea accesible.

## pyproject.toml (gestión moderna)
- Define el sistema de build moderno (`setuptools` + `wheel`) y metadatos del proyecto para Python >= 3.11.
- Las dependencias de runtime se mantienen en `requirements.txt` para flujos sencillos (pip/Docker). El proyecto está listo para migrar a dependencias declarativas en `pyproject.toml` si se desea en el futuro.


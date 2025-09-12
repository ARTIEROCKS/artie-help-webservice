# Help Webservice

Flask service for inference of a help recommendation model (help_model.keras), with prediction and health endpoints, ready for Python 3.11 and Docker.

## Features
- Python 3.11, Flask 3.
- TensorFlow/Keras 2.15 with custom layers registered in `lib/keras_custom_layers.py`.
- Endpoints:
  - `GET /health` — returns OK only when the model `model/help_model.keras` was successfully loaded at startup.
  - `POST /api/v1/help-model/predict` — receives student interactions and returns whether help should be shown.
- Dockerfile includes a native HEALTHCHECK that probes `/health`.

## Requirements
- Model file at `model/help_model.keras`.
- Python 3.11 (for local runs) or Docker.
- For the prediction endpoint: access to MongoDB and environment variables set.

### Environment variables (MongoDB)
The prediction flow reads/writes interactions via Mongo using `repository/db.py` and requires:
- `APP_MONGO_HOST`
- `APP_MONGO_PORT`
- `APP_MONGO_USER`
- `APP_MONGO_PASS`
- `APP_MONGO_DB`

Example (bash):
```bash
export APP_MONGO_HOST=localhost
export APP_MONGO_PORT=27017
export APP_MONGO_USER=myuser
export APP_MONGO_PASS=mypass
export APP_MONGO_DB=artie
```

## Quick start

### A) Docker (recommended)
1) Build the image
```bash
docker build -t help-webservice .
```
2) Run the container
- Health-only (Mongo not required):
```bash
docker run -d --name help-webservice -p 8000:8000 help-webservice
```
- With prediction (requires Mongo and env vars):
```bash
docker run -d --name help-webservice \
  -p 8000:8000 \
  -e APP_MONGO_HOST=host.docker.internal \
  -e APP_MONGO_PORT=27017 \
  -e APP_MONGO_USER=myuser \
  -e APP_MONGO_PASS=mypass \
  -e APP_MONGO_DB=artie \
  help-webservice
```
3) Probe health
```bash
curl -i http://localhost:8000/health
```
You should get `200 {"status":"ok"}` once the model is loaded.

Container health status (Docker):
```bash
docker ps --filter name=help-webservice --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

### B) Local (venv)
1) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
Notes:
- On macOS, `tensorflow-macos` (and `tensorflow-metal` on Apple Silicon) will be installed automatically via markers in `requirements.txt`.
- On Linux, `tensorflow==2.15.0` will be installed.

3) Run the service
```bash
python app.py
```
4) Probe health
```bash
curl -i http://localhost:8000/health
```

### Debug mode (local)
Use Flask’s debug server with live reload:
```bash
export FLASK_APP=app.py
flask --app app --debug run -h 0.0.0.0 -p 8000
```
If you see the model loading twice on startup (due to the reloader), run without the reloader:
```bash
flask --app app --debug run --no-reload -h 0.0.0.0 -p 8000
```

## Prediction API
- URL: `POST /api/v1/help-model/predict`
- Content-Type: `application/json`
- Body: list of interaction objects. Minimal fields used by the preprocessor: `student.id` (or `student._id`), `exercise.id` (or `exercise._id`), `lastLogin`, `dateTime`.

Example:
```bash
curl -s -X POST http://localhost:8000/api/v1/help-model/predict \
  -H 'Content-Type: application/json' \
  -d '[{
    "student": {"id": "S1"},
    "exercise": {"id": "E1"},
    "lastLogin": "2024-03-01T10:00:00.000000",
    "dateTime":  "2024-03-01T10:00:05.000000",
    "requestHelp": false
  }]'
```
Response (stringified JSON with a `body` wrapper):
```
{"body": {"message": "OK", "object": 0}}
```
`object` is 1 when help should be shown; otherwise 0.

## Project structure
```
app.py                     # Flask app, endpoints /health and /predict
service/
  preprocess.py            # Data transformations
  model.py                 # Model loading and prediction
  queue_service.py         # Fetch/update interactions in Mongo
repository/
  db.py                    # Mongo client and ops (reads env vars)
lib/
  keras_custom_layers.py   # Registered custom Keras layers/functions
model/
  help_model.keras         # Default model
  selectedfeatures.csv     # CSV with expected input columns
```

## Compatibility notes
- Python 3.11.
- TensorFlow/Keras 2.15.
- `requirements.txt` pins versions that avoid problematic native builds on Py3.11 (`grpcio==1.62.2`, `PyYAML==6.0.2`, `wrapt==1.14.1`).
- In `app.py`, custom objects are imported from `lib/keras_custom_layers.py` so model deserialization works without providing a `custom_objects` dict at call sites.

## Troubleshooting
- `ImportError: cannot import name 'formatargspec' from 'inspect'` — use `wrapt==1.14.1` (already pinned) instead of older versions.
- Build failures for `grpcio` or `PyYAML` — the pinned versions ship wheels for Py3.11.
- `/health` returns 503 — ensure `model/help_model.keras` exists and is readable; check container logs or console.
- Prediction fails due to DB connection — export Mongo env vars and ensure the instance is reachable.

## pyproject.toml (modern build)
- Declares the modern build system (`setuptools` + `wheel`) and basic project metadata for Python >= 3.11.
- Runtime dependencies remain in `requirements.txt` for simple pip/Docker flows. You can migrate to declarative dependencies in `pyproject.toml` later if desired.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

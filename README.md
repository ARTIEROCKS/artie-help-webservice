# Help Webservice

FastAPI service that serves a trained HelpModel (Keras) and, if available, an attention submodel. It exposes health and prediction endpoints on port 8000.

## What it does
- Loads `model/help_model.keras` at startup and serves time-step predictions for sequences of interactions.
- Optionally loads `model/help_model_attention.keras` and returns top-k attention steps alongside the prediction.
- Health endpoint to check readiness (and whether the attention model is loaded).

## Tech
- Python 3.11
- FastAPI + Uvicorn
- TensorFlow/Keras 2.18

## Endpoints
- GET `/health`
  - Returns `{ "status": "ok" | "model_not_loaded", "attention_model": "loaded" | "absent" }`.

- POST `/api/v1/help-model/predict`
  - Body: a JSON array of interaction objects. Minimal fields used are inside `student`, `exercise.skills`, `exercise.level`, `solutionDistance.totalDistance`, `secondsHelpOpen`, and timestamps `dateTime` and `lastLogin`.
  - Example body:
    ```json
    [
      {
        "student": {"_id": "6202b0907f8f0c5052ac8fbb", "gender": 1, "motherTongue": 2, "age": 38, "competence": 1, "motivation": 0},
        "exercise": {
          "_id": "60b50dffe2d2f2195608cedb",
          "skills": [
            {"name": "Paralelismo", "score": 0.0},
            {"name": "Pensamiento lógico", "score": 0.0},
            {"name": "Control de flujo", "score": 0.66},
            {"name": "Interactividad con el usuario", "score": 0.33},
            {"name": "Representación de la información", "score": 0.33},
            {"name": "Abstracción", "score": 0.0},
            {"name": "Sincronización", "score": 0.67}
          ],
          "validSolution": 0,
          "isEvaluation": true,
          "level": 1
        },
        "solutionDistance": {"totalDistance": 64.625},
        "dateTime": "2021-06-08 11:08:36.121000",
        "secondsHelpOpen": 0.0,
        "finishedExercise": false,
        "validSolution": 0,
        "grade": 0.26365348399246713,
        "lastLogin": "2021-06-08 12:06:50",
        "aptedDistance": 0.0
      }
    ]
    ```
  - Response:
    ```json
    {
      "message": "OK",
      "threshold": 0.5,
      "help_needed": true,
      "last_probability": 0.73,
      "sequence_probabilities": [0.21, 0.34, ..., 0.73],
      "attention": {
        "available": true,
        "top_k": [{"t": 58, "w": 0.091}, {"t": 57, "w": 0.089}, {"t": 59, "w": 0.071}],
        "seq_len": 60
      }
    }
    ```
    When the attention model is not present, `attention.available` will be `false` and the other fields are omitted.

## Environment variables
- `HELP_MODEL_PATH` (default `model/help_model.keras`): path to the main model.
- `HELP_ATTENTION_MODEL_PATH` (default `model/help_model_attention.keras`): path to the attention model (optional).
- `HELP_MODEL_THRESHOLD` (default `0.5`): threshold to turn the last probability into `help_needed`.
- `HELP_ATTENTION_TOPK` (default `5`): number of top attention steps returned in `attention.top_k`.

## Run locally
1) Create venv and install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2) Run the service
```bash
python app.py
# or explicitly
uvicorn app:app --host 0.0.0.0 --port 8000
```
3) Probe health
```bash
curl -s http://localhost:8000/health | jq
```

## Docker
Build the image:
```bash
docker build -t help-webservice .
```
Run the container:
```bash
docker run -d --name help-webservice -p 8000:8000 \
  -e HELP_MODEL_PATH=model/help_model.keras \
  -e HELP_ATTENTION_MODEL_PATH=model/help_model_attention.keras \
  help-webservice
```
Healthcheck (Dockerfile includes one that probes `/health`). You can also check manually:
```bash
curl -s http://localhost:8000/health | jq
```

## Notes
- Input features expected by the model (15):
  - student_sex, student_mother_tongue, student_age, student_competence,
  - exercise_skill_parallelism, exercise_skill_logical_thinking, exercise_skill_flow_control,
  - exercise_skill_user_interactivity, exercise_skill_information_representation,
  - exercise_skill_abstraction, exercise_skill_synchronization,
  - exercise_level, solution_distance_total_distance, seconds_help_open, total_seconds.
- The API computes `total_seconds` on the fly relative to the first event timestamp in the received sequence.
- Columns related to APTED are ignored if present in the payload.

## Troubleshooting
- `/health` returns `model_not_loaded`: ensure `HELP_MODEL_PATH` points to a valid Keras model file inside the container/working dir.
- Attention not available: ensure `HELP_ATTENTION_MODEL_PATH` exists and is loadable; otherwise `attention.available` will be `false`.
- Prediction input errors: verify the body is a non-empty JSON array and timestamps follow `YYYY-mm-dd HH:MM:SS[.ffffff]`.

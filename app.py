import os
from datetime import datetime
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
import tensorflow as tf

# Model paths and runtime parameters
MODEL_PATH = os.getenv("HELP_MODEL_PATH", "model/help_model.keras")
ATTENTION_MODEL_PATH = os.getenv("HELP_ATTENTION_MODEL_PATH", "model/help_model_attention.keras")
THRESHOLD = float(os.getenv("HELP_MODEL_THRESHOLD", "0.5"))
ATTENTION_TOPK = int(os.getenv("HELP_ATTENTION_TOPK", "5"))

# Exact order of expected input features (15, aligned with training)
FEATURE_ORDER = [
    "student_sex",
    "student_mother_tongue",
    "student_age",
    "student_competence",
    "exercise_skill_parallelism",
    "exercise_skill_logical_thinking",
    "exercise_skill_flow_control",
    "exercise_skill_user_interactivity",
    "exercise_skill_information_representation",
    "exercise_skill_abstraction",
    "exercise_skill_synchronization",
    "exercise_level",
    "solution_distance_total_distance",
    "seconds_help_open",
    "total_seconds",
]

# Columns related to APTED not used by the model
APTED_COLUMNS = ["apted_distance", "tree_grade"]

# Mapping from skill display names to feature columns
SKILL_NAME_MAP = {
    "Paralelismo": "exercise_skill_parallelism",
    "Pensamiento lógico": "exercise_skill_logical_thinking",
    "Control de flujo": "exercise_skill_flow_control",
    "Interactividad con el usuario": "exercise_skill_user_interactivity",
    "Representación de la información": "exercise_skill_information_representation",
    "Abstracción": "exercise_skill_abstraction",
    "Sincronización": "exercise_skill_synchronization",
}

app = FastAPI(title="HelpModel WebService", version="1.0.0")

# Load main model on startup
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load main model on startup: {e}")

# Load attention model on startup if present
attention_model = None
try:
    if os.path.exists(ATTENTION_MODEL_PATH):
        attention_model = tf.keras.models.load_model(ATTENTION_MODEL_PATH, compile=False, safe_mode=False)
    else:
        print("[INFO] Attention model not found; attention will be omitted in responses")
except Exception as e:
    attention_model = None
    print(f"[WARN] Failed to load attention model: {e}")


@app.get("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    att = "loaded" if attention_model is not None else "absent"
    return {"status": status, "attention_model": att}


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Try to parse a datetime string with or without microseconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except Exception:
            continue
    # Fallback with pandas (more flexible)
    try:
        return pd.to_datetime(dt_str).to_pydatetime()
    except Exception:
        return None


def transform_sequence(payload: List[Dict[str, Any]]) -> np.ndarray:
    """Transform a list of interaction objects into model-ready tensor of shape (1, T, F)."""
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("Body must be a non-empty array of interactions")

    rows = []
    # Ensure chronological order by dateTime
    sorted_payload = sorted(payload, key=lambda x: x.get("dateTime", ""))

    # Compute total_seconds relative to the first action timestamp
    first_dt = None
    for item in sorted_payload:
        dt = parse_datetime(item.get("dateTime")) if item.get("dateTime") else None
        if dt is not None:
            first_dt = dt
            break

    for item in sorted_payload:
        row = {col: 0.0 for col in FEATURE_ORDER}

        # Student fields: gender -> sex; motherTongue, age, competence
        student = item.get("student", {}) or {}
        if "gender" in student:
            row["student_sex"] = float(student.get("gender") or 0)
        if "motherTongue" in student:
            row["student_mother_tongue"] = float(student.get("motherTongue") or 0)
        if "age" in student:
            row["student_age"] = float(student.get("age") or 0)
        if "competence" in student:
            row["student_competence"] = float(student.get("competence") or 0)
        # motivation is not used as a final feature

        # Exercise: map skills and level
        exercise = item.get("exercise", {}) or {}
        skills = exercise.get("skills", []) or []
        for s in skills:
            name = s.get("name")
            score = float(s.get("score") or 0.0)
            col = SKILL_NAME_MAP.get(name)
            if col:
                row[col] = score
        if "level" in exercise:
            row["exercise_level"] = float(exercise.get("level") or 0)

        # ARTIE distances: totalDistance
        solution_distance = item.get("solutionDistance", {}) or {}
        if "totalDistance" in solution_distance:
            row["solution_distance_total_distance"] = float(solution_distance.get("totalDistance") or 0.0)

        # seconds_help_open
        if "secondsHelpOpen" in item:
            row["seconds_help_open"] = float(item.get("secondsHelpOpen") or 0.0)

        # total_seconds relative to the first action
        current_dt = parse_datetime(item.get("dateTime")) if item.get("dateTime") else None
        if first_dt is not None and current_dt is not None:
            row["total_seconds"] = max(0.0, (current_dt - first_dt).total_seconds())
        else:
            row["total_seconds"] = 0.0

        # Drop APTED-related columns if present (not part of FEATURE_ORDER)
        for c in APTED_COLUMNS:
            if c in row:
                row.pop(c, None)

        # Keep the expected columns in fixed order
        rows.append([row[c] for c in FEATURE_ORDER])

    # Output shape (1, T, F)
    X = np.array(rows, dtype=np.float32)
    X = np.expand_dims(X, axis=0)
    return X


def _topk_weights(w: np.ndarray, k: int) -> List[Dict[str, float]]:
    """Return the top-k attention weights as a list of {t, w}."""
    if w.size == 0:
        return []
    k = max(1, min(k, w.size))
    idx = np.argpartition(-w, k - 1)[:k]
    idx = idx[np.argsort(-w[idx])]
    return [{"t": int(i), "w": float(w[i])} for i in idx]


@app.post("/api/v1/help-model/predict")
async def predict(request: Request):
    global model, attention_model
    if model is None:
        # Retry loading if it failed on startup
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load main model: {e}")

    try:
        payload = await request.json()
        X = transform_sequence(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        # Time-step prediction (model trained with return_sequences=True)
        preds = model.predict(X, verbose=0).astype(float).reshape(-1)
        # Use the last probability as the current decision
        last_prob = float(preds[-1]) if preds.size > 0 else 0.0
        help_needed = bool(last_prob >= THRESHOLD)

        # Compute attention if the submodel is available
        attention = {"available": False}
        if attention_model is None and os.path.exists(ATTENTION_MODEL_PATH):
            try:
                attention_model = tf.keras.models.load_model(ATTENTION_MODEL_PATH, compile=False, safe_mode=False)
            except Exception as e:
                attention_model = None
                print(f"[WARN] On-demand attention model load failed: {e}")

        if attention_model is not None:
            try:
                att = attention_model.predict(X, verbose=0)
                # Normalize to shape (T,)
                if att.ndim == 3 and att.shape[-1] == 1:
                    att = att[0, :, 0]
                elif att.ndim == 2:
                    att = att[0, :]
                else:
                    att = np.array([], dtype=np.float32)
                attention = {
                    "available": True,
                    "top_k": _topk_weights(att.astype(float), ATTENTION_TOPK),
                    "seq_len": int(att.shape[0])
                }
            except Exception as e:
                print(f"[WARN] Failed to compute attention: {e}")

        return {
            "message": "OK",
            "body": {
                "threshold": THRESHOLD,
                "help_needed": help_needed,
                "last_probability": last_prob,
                "sequence_probabilities": preds.tolist(),
                "attention": attention,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

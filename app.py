# --- Archivo: tu servicio Flask (el que expone '/api/v1/help-model/predict')
from flask import Flask, request
from flask_cors import CORS
from service import preprocess, queue_service
import logging, os, threading, numpy as np, pandas as pd
from keras import models as kmodels
from lib.keras_custom_layers import (
    compute_mask_layer,
    squeeze_last_axis_func,
    mask_attention_scores_func,
    apply_attention_func,
    MaskedRepeatVector,
    AttentionLayer,
    func,  # <- alias necesario para deserializar la Lambda
    # Nuevas capas para reemplazar Lambdas problemáticas
    SqueezeLastAxisLayer,
    ComputeMaskLayer,
    MaskAttentionScoresLayer,
    ApplyAttentionLayer,
)

from keras.saving import register_keras_serializable

# Registra funciones y capas personalizadas (si no estaban decoradas ya)
compute_mask_layer = register_keras_serializable(package="Custom", name="compute_mask_layer")(compute_mask_layer)
squeeze_last_axis_func = register_keras_serializable(package="Custom", name="squeeze_last_axis_func")(squeeze_last_axis_func)
mask_attention_scores_func = register_keras_serializable(package="Custom", name="mask_attention_scores_func")(mask_attention_scores_func)
apply_attention_func = register_keras_serializable(package="Custom", name="apply_attention_func")(apply_attention_func)
MaskedRepeatVector = register_keras_serializable(package="Custom", name="MaskedRepeatVector")(MaskedRepeatVector)
AttentionLayer = register_keras_serializable(package="Custom", name="AttentionLayer")(AttentionLayer)

# Alias retrocompatible para la Lambda guardada como 'func'
# Cambia el destino si tu Lambda usaba otra función.
func = squeeze_last_axis_func
func.__name__ = "func"
func = register_keras_serializable(package="Custom", name="func")(func)

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/help_model.keras"
SELECTED_FEATURES_PATH = "model/selectedfeatures.csv"  # Añadir la variable que faltaba
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_ERROR = None

def _get_custom_objects():
    # Incluye 'func' explícitamente y las nuevas capas
    return {
        "compute_mask_layer": compute_mask_layer,
        "squeeze_last_axis_func": squeeze_last_axis_func,
        "mask_attention_scores_func": mask_attention_scores_func,
        "apply_attention_func": apply_attention_func,
        "MaskedRepeatVector": MaskedRepeatVector,
        "AttentionLayer": AttentionLayer,
        "func": func,
        # Nuevas capas para reemplazar Lambdas problemáticas
        "SqueezeLastAxisLayer": SqueezeLastAxisLayer,
        "ComputeMaskLayer": ComputeMaskLayer,
        "MaskAttentionScoresLayer": MaskAttentionScoresLayer,
        "ApplyAttentionLayer": ApplyAttentionLayer,
    }

def _load_model_once():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return
    if not os.path.exists(MODEL_PATH):
        _MODEL_ERROR = f"Model file not found at {MODEL_PATH}"
        return
    with _MODEL_LOCK:
        if _MODEL is not None:
            return
        try:
            _MODEL = kmodels.load_model(
                MODEL_PATH,
                custom_objects=_get_custom_objects(),
                compile=False,
                safe_mode=False,  # necesario para código/funciones personalizados
            )
            _MODEL_ERROR = None
        except Exception as ex:
            _MODEL = None
            _MODEL_ERROR = str(ex)

def _read_selected_features():
    if not os.path.exists(SELECTED_FEATURES_PATH):
        return None
    df_feat = pd.read_csv(SELECTED_FEATURES_PATH)
    first_col = df_feat.columns[0]
    return df_feat[first_col].astype(str).tolist()

def _prepare_input_for_model(x):
    _load_model_once()
    if _MODEL is None:
        raise RuntimeError(f"Model not ready: {_MODEL_ERROR}")
    selected = _read_selected_features()
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        if selected:
            for col in selected:
                if col not in df.columns:
                    df[col] = 0.0
            X = df[selected].to_numpy(dtype=np.float32)
        else:
            X = df.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(x, dtype=np.float32)
    # Asegura dimensión batch
    try:
        input_rank = len(tuple(_MODEL.inputs[0].shape))
    except Exception:
        input_rank = X.ndim + 1
    if X.ndim == (input_rank - 1):
        X = np.expand_dims(X, axis=0)
    while X.ndim < (input_rank - 1):
        X = np.expand_dims(X, axis=0)
    return X

# Carga temprana
_load_model_once()

@app.route('/health', methods=['GET'])
def health():
    if _MODEL is not None and _MODEL_ERROR is None:
        return {"status": "ok"}, 200
    return {"status": "not_ready", "error": _MODEL_ERROR}, 503

@app.route('/api/v1/help-model/predict', methods=['POST'])
def predict():
    if request.method != "POST":
        return None
    logging.info("New help model prediction requested by API REST")

    error_message = ""
    prediction_int = None
    try:
        elements = request.data
        student_interactions, client = queue_service.get_student_interactions(elements)
    except Exception as ex:
        error_message = "Error: Predict - get_student_interactions: " + str(ex)
        student_interactions = None

    try:
        if student_interactions is not None:
            df = preprocess.data_transformation(student_interactions["interactions"])
        else:
            df = None
    except Exception as ex:
        error_message = (error_message + " | " if error_message else "") + "Error: Predict - data_transformation: " + str(ex)
        df = None

    try:
        if df is not None:
            X = _prepare_input_for_model(df)
            y_pred = _MODEL.predict(X, verbose=0)
            prediction_int = np.rint(y_pred)
    except Exception as ex:
        error_message = (error_message + " | " if error_message else "") + "Error: Predict - prediction: " + str(ex)

    if prediction_int is not None:
        help_needed = 1 in prediction_int
        return "{\"body\": {\"message\": \"OK\", \"object\": " + str(int(help_needed)) + "}}"
    else:
        return "{\"body\": {\"message\": \"ERROR\", \"object\": \"" + error_message + "\"}}"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)

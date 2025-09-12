from flask import Flask
from flask import request
from flask_cors import CORS
from service import preprocess, model, queue_service  # restored queue_service import
import numpy as np
import logging
import tensorflow as tf
import os
from lib.keras_custom_layers import *

app = Flask(__name__)
CORS(app)

# --- Model readiness state (English comments) ---
MODEL_PATH = "model/help_model.keras"
ATTENTION_MODEL_PATH = "model/help_model_attention.keras"

_model_loaded = False
_model_error = None

_attention_model_loaded = False
_attention_model_error = None


def _load_model_once():
    """Load the model once at startup to declare readiness for /health."""
    global _model_loaded, _model_error
    if _model_loaded:  # already loaded
        return
    try:
        if not os.path.exists(MODEL_PATH):
            _model_error = f"Model file not found at {MODEL_PATH}"
            return

        # Build custom objects map (in case registry is not enough)
        custom_objects = {
            'compute_mask_layer': compute_mask_layer,
            'squeeze_last_axis_func': squeeze_last_axis_func,
            'mask_attention_scores_func': mask_attention_scores_func,
            'apply_attention_func': apply_attention_func,
            'MaskedRepeatVector': MaskedRepeatVector,
            'AttentionLayer': AttentionLayer,
        }

        # Load once for readiness (discard instance; actual prediction loads in service/model)
        tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)

        _model_loaded = True
        _model_error = None
    except Exception as ex:
        _model_error = str(ex)
        _model_loaded = False


# Early load at import time
_load_model_once()


@app.route('/health', methods=['GET'])
def health():
    """Return 200 only when model was successfully loaded at startup."""
    if _model_loaded:
        return {"status": "ok"}, 200
    # Not ready yet / failed
    return {"status": "not_ready", "error": _model_error}, 503


@app.route('/api/v1/help-model/predict', methods=['POST'])
def predict():
    if request.method == "POST":

        logging.info("New help model prediction requested by API REST")
        print("New help model prediction requested by API REST")

        error_message = ""
        student_interactions = None
        df = None
        prediction_int = None

        try:

            logging.info("Predict - get_student_interactions")
            print("Predict - get_student_interactions")

            # Get the data and searches for the student interactions
            elements = request.data
            student_interactions, client = queue_service.get_student_interactions(elements)
        except Exception as ex:
            error_message = "Error: Predict - get_student_interactions: " + str(ex)
            logging.error(error_message)
            print(error_message)

        try:
            if student_interactions is not None:
                logging.info("Predict - data_transformation")
                print("Predict - data_transformation")

                # Once we have the interactions of the student, we transform the data to get a valid array
                df = preprocess.data_transformation(student_interactions["interactions"])
        except Exception as ex:
            error_message = error_message + " | Error: Predict - data_transformation: " + str(ex)
            logging.error(error_message)
            print(error_message)

        try:
            if df is not None:
                logging.info("Predict - prediction")
                print("Predict - prediction")

                # Predicts the output
                prediction = model.predict("model/help_model.keras", "model/selectedfeatures.csv", df)

                # Round the prediction to integers
                prediction_int = np.rint(prediction)
        except Exception as ex:
            error_message = error_message + " | Error: Predict - prediction: " + str(ex)
            logging.error(error_message)
            print(error_message)

        if prediction_int is not None:

            # Searches if the help must be shown
            help_needed = 1 in prediction_int
            help_object = "{\"body\": {\"message\": \"OK\", \"object\": " + str(int(help_needed)) + "}}"

        else:
            # If there are no predictions, we return the errors
            help_object = "{\"body\": {\"message\": \"ERROR\", \"object\": \"" + error_message + "\"}}"

        logging.info("API Rest response: " + str(help_object))
        print("API Rest response: " + str(help_object))

        return help_object
    return None


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)

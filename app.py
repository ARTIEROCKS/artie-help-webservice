from flask import Flask
from flask import request
from flask_cors import CORS
from service import preprocess, model, queue_service
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/help-model/predict', methods=['POST'])
def predict():
    if request.method == "POST":

        error_message = ""
        student_interactions = None
        df = None
        prediction_int = None

        try:
            # Get the data and searches for the student interactions
            elements = request.data
            student_interactions, client = queue_service.get_student_interactions(elements)
        except:
            error_message = "Error: Predict - get_student_interactions"
            print("Error: Predict - get_student_interactions")

        try:
            if student_interactions is not None:
                # Once we have the interactions of the student, we transform the data to get a valid array
                df = preprocess.data_transformation(student_interactions["interactions"])
        except:
            error_message = error_message + " | Error: Predict - data_transformation"
            print("Error: Predict - data_transformation")

        try:
            if df is not None:
                # Predicts the output
                prediction = model.predict("model/help_model.h5", "model/selectedfeatures.csv", df)

                # Round the prediction to integers
                prediction_int = np.rint(prediction)
        except:
            error_message = error_message + " | Error: Predict - prediction"
            print("Error: Predict - prediction")

        if prediction_int is not None:

            # Searches if there if the help must be shown
            help = 1 in prediction_int
            help_object = "{\"body\": {\"message\": \"OK\", \"object\": " + str(int(help)) + "}}"

        else:
            # If there are no predictions, we return the errors
            help_object = "{\"body\": {\"message\": \"ERROR\", \"object\": \"" + error_message + "\"}}"

        return help_object


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port="5000")

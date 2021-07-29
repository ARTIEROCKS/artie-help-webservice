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

        # Get the data and searches for the student interactions
        elements = request.data
        student_interactions = queue_service.get_student_interactions(elements)

        # Once we have the interactions of the student, we transform the data to get a valid array
        df = preprocess.data_transformation(student_interactions["interactions"])

        # Predicts the output
        prediction = model.predict("model/help_model.h5", "model/selectedfeatures.csv", df)

        # Round the prediction to integers
        prediction_int = np.rint(prediction)

        # Searches if there if the help must be shown
        help = 1 in prediction_int
        help_object = "{show_help: " + str(int(help)) + "}"
        print(help_object)

        return help_object


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port="5000")

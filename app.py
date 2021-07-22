from flask import Flask
from flask import request
from flask_cors import CORS
import preprocess
import model
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/help-model/predict', methods=['POST'])
def predict():
    if request.method == "POST":

        # Get the data and preprocess the data
        elements = request.data
        df = preprocess.data_transformation(elements)

        # Predicts the output
        prediction = model.predict("model/help_model.h5", "model/selectedfeatures.csv", df)
        print(prediction)

        # Round the prediction to integers
        prediction_int = np.rint(prediction)

        # Searches if there if the help must be shown
        help = 1 in prediction_int
        help_object = "{show_help: " + str(int(help)) + "}"
        print(help_object)

        return help_object


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port="80")

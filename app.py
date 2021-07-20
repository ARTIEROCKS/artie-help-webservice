from flask import Flask
from flask import request
from flask_cors import CORS
import preprocess
import model

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/help-model/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        elements = request.data
        df = preprocess.data_transformation(elements)
        prediction = model.predict("model/help_model.h5", "model/selectedfeatures.csv", df)
        return prediction


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="80")

import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Function to preprocess the information that will be loaded in the model
def preprocess(df, selected_features_file):

    # 1 - Loads the csv with the selected features
    selected_features = pd.read_csv(selected_features_file)
    selected_features_columns = list(selected_features.columns)

    # 2- Deletes all the unnecessary columns
    df.drop(columns=[col for col in df if col not in selected_features_columns], inplace=True)

    return df


# Function to predict the class
def predict(model_url, selected_features_file, x):
    data = preprocess(x, selected_features_file)
    data_np = data.to_numpy()
    data_np = np.reshape(data_np, (1, data_np.shape[0], data_np.shape[1]))
    model = load_model(model_url)
    return model.predict(data_np, batch_size=1)

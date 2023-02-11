from feature_extraction import normalize_volume, extract_features
from sklearn.preprocessing import StandardScaler
import numpy as np


def file_to_prediction(file_path):

    # Run through the functions from feature_extraction to return list of features
    y_norm, sr = normalize_volume(file_path)
    feature_list = extract_features(y_norm, sr)

    # Normalize the list of features
    #
    # ALEX: I COMMENTED OUT THE BELOW CODE FOR NOW AS IT
    # DOESN'T WORK
    # scaled_feature_list = list_to_scaled_list(feature_list)

    # Here is where the model can be called to predict the data

    # This is just for demo purposes
    return np.arange(89)



def list_to_scaled_list(feature_list):
    # Use standard scaler from sklearn library
    scaler = StandardScaler()

    scaled_feature_list = scaler.fit_transform(feature_list)

    return scaled_feature_list

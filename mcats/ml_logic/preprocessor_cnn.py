import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from mcats.ml_logic.registry import save_preprocessor, load_preprocessor
import os

def preprocess_cnn(X,y):

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the encoder to one-hot encode y
    y = encoder.fit_transform(y).toarray()

    # Add a new axis to X data for CNN
    X = X[...,np.newaxis]

    save_preprocessor(encoder)

    return X, y

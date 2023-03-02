import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

def extract_json_data(path):
    X = []
    y = []
    target_shape = (13, 130)

    # List of file names to be merged
    path = path
    file_names = ['country.json', 'classical.json', 'metal.json', 'hiphop.json', 'electronic.json', 'pop.json', 'reggae.json', 'rock.json']

    # Iterate through each file name and load the JSON data
    for file_name in file_names:
        full_path = os.path.join(path,file_name)
        with open(full_path, 'r') as file:
            data = json.load(file)

        # Iterate over each song and extract the MFCCs
        songs = data.keys()
        for song in songs:
            i=0
            if f'seg_{i}' in data[song]:
                for i in range(10):
                    # Create a NumPy array of all 13 MFCCs for each segment
                    mfcc_array = np.array(data[song][f'seg_{i}'])
                    pad_width = [(0, max(0, target_shape[i] - mfcc_array.shape[i])) for i in range(len(target_shape))]
                    padded_mfcc = np.pad(mfcc_array, pad_width=pad_width, mode='constant', constant_values=0)
                    X.append(padded_mfcc)
                    y.append(data[song]['genre'])

    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape y to a column vector
    y = np.array(y).reshape(-1, 1)

    # Add a new axis to X data for CNN
    # X = X[...,np.newaxis]

    return X, y

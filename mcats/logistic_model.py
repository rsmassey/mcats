import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from mcats.ml_logic.params import (
    LOCAL_DATA_PATH,
    DATASET_SIZE
)


def train_and_classify(song_features, source_type = 'train'):

    #Import the data from the csv file
    csv_path = f'{LOCAL_DATA_PATH}/{source_type}_{DATASET_SIZE}.csv'
    data = pd.read_csv(csv_path)

    #Dropping the first column of the csv file
    #The first column contains the name of the audio file.
    # Irrelevant for the task.
    data  = data.drop("filename", axis=1)

    #Setting up the featurs and the target variables
    y = np.array(data['label'])
    X = np.array(data.drop('label', axis=1))

    #Initialising a Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    #Split the data between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=42)

    #Cross validate the data
    cv_results = cross_validate(model, X_train, y_train, cv=5)

    #Getting a score
    accuracy = cv_results['test_score'].mean()

    #Training the data
    model.fit(X_train, y_train)

    X_to_classify = [song_features]
    classification = model.predict(X_to_classify)[0]

    #Print the classification of the input song
    return classification

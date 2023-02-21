

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.preprocessing import StandardScaler
from mcats.ml_logic.registry import save_preprocessor, load_preprocessor


def preprocess_features(X: pd.DataFrame, val: bool, pred: bool) -> np.ndarray:

    # Preprocessing X for prediction
    if pred:
        scaler = load_preprocessor()
        X = np.array(X).T
        X = np.array(X).reshape(1, 89)
        X_standardized = scaler.transform(X)


    # Preprocessing X for model validation
    elif val:
        scaler = load_preprocessor()
        X_without_genre = X.iloc[:, :-1].copy()
        X_standardized = pd.DataFrame(scaler.transform(X),
                               columns=X.columns, index=X.index)
        X_standardized = X_standardized.to_numpy()

    # Preprocessing X for model training
    else:
        # Use standard scaler from sklearn library
        scaler = StandardScaler()
        # Remove the genre column and standardize the features
        X_without_genre = X.iloc[:, :-1].copy()
        X_standardized = pd.DataFrame(scaler.fit_transform(X),
                               columns=X.columns, index=X.index)
        X_standardized = X_standardized.to_numpy()
        save_preprocessor(scaler)

    print("\n✅ X_processed")

    return X_standardized

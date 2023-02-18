

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.preprocessing import StandardScaler


def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    # Use standard scaler from sklearn library
    scaler = StandardScaler()

    # Remove the genre column and standardize the features
    X_without_genre = X.iloc[:, :-1].copy()
    X_standardized = pd.DataFrame(scaler.fit_transform(X),
                               columns=X.columns, index=X.index)

    X_standardized = X_standardized.to_numpy()

    print("\n✅ X_processed")

    return X_standardized

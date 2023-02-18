
from colorama import Fore, Style
from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate


def train_model(model,
                X: np.ndarray,
                y: np.ndarray) -> Tuple[SVC, dict]:
    """
    Fit model and return a the tuple (fitted_model, accuracy)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Training the data
    model.fit(X, y)

    #Getting a score
    #accuracy = model.score(X, y)

    #Cross-validation
    cv_results = cross_validate(X,
                              y,
                              model,
                              cv=5)

    accuracy = cv_results['test_score'].mean()

    return model, accuracy


def initialize_model(kernel, C) -> SVC:
    """
    Initialize the Logistic Regression
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    model = SVC(kernel=kernel, C=C)

    print("\n✅ model initialized")

    return model

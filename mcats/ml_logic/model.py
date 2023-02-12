
from colorama import Fore, Style
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def train_model(model,
                X: np.ndarray,
                y: np.ndarray,
                cv=5) -> Tuple[LogisticRegression, dict]:
    """
    Fit model and return a the tuple (fitted_model, accuracy)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Cross validate the data
    cv_results = cross_validate(model, X, y, cv=cv)

    #Getting a score
    accuracy = cv_results['test_score'].mean()

    #Training the data
    model.fit(X, y)


    return model, accuracy


def initialize_model(max_iter) -> LogisticRegression:
    """
    Initialize the Logistic Regression
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    model = LogisticRegression(max_iter=max_iter)

    print("\n✅ model initialized")

    return model

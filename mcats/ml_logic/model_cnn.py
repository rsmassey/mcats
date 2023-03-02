
from colorama import Fore, Style
from typing import Tuple
import numpy as np
from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, models
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")


def initialize_model(X: np.ndarray) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    model = models.Sequential()

    input_shape = (13, 130, 1)

    ### First Convolution, MaxPooling, and Normalization
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(32, (2,2), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(32, (2,2), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    ### Flattening
    model.add(layers.Flatten())

    ### One more Dense Layer
    model.add(layers.Dense(64, activation='relu'))

    ### A Dropout layer to avoid overfitting
    model.add(layers.Dropout(0.3))

    ### Last layer - Classification Layer with 10 outputs corresponding to 10 digits
    model.add(layers.Dense(8, activation='softmax'))


    print("\n✅ model initialized")

    return model


def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Neural Network
    """

    ### Model compilation
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("\n✅ model compiled")

    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=32,
                patience=10,
                validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Setup early stopping
    es = EarlyStopping(patience = patience, monitor='accuracy')

    #Training the data
    history = model.fit(X,
              y,
              validation_split=validation_split,
              validation_data=validation_data,
              batch_size = batch_size,
              epochs = 50,
              callbacks = [es],
              verbose = 1)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history



def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    accuracy = metrics["accuracy"]
    # mae = metrics["mae"]

    # print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    print(f"\n✅ model evaluated: accuracy {round(accuracy, 2)}")

    return metrics

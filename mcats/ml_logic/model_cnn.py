
from colorama import Fore, Style
from typing import Tuple
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam


def train_model(model,
                X: np.ndarray,
                y: np.ndarray) -> Tuple[SVC, dict]:
    """
    Fit model and return a the tuple (fitted_model, accuracy)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Seetup early stopping
    es = EarlyStopping(patience = 10, monitor='accuracy')

    #Training the data
    model.fit(X,
              y,
              batch_size = 32,
              epochs = 50,
              callbacks = [es],
              verbose = 1)

    #Getting a score
    #accuracy = model.score(X, y)

    return model, accuracy


def initialize_model(kernel, C) -> SVC:
    """
    Initialize the Logistic Regression
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

    ### Model compilation
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("\n✅ model initialized")

    return model

import numpy as np
import pandas as pd
import os

from colorama import Fore, Style

from mcats.ml_logic.data import clean_data, get_chunk, save_chunk
from mcats.ml_logic.model import  train_model, initialize_model
from mcats.ml_logic.model_cnn import  train_model, initialize_model, compile_model
from mcats.ml_logic.params import CHUNK_SIZE, DATASET_SIZE, VALIDATION_DATASET_SIZE
from mcats.ml_logic.preprocessor import preprocess_features
from mcats.ml_logic.registry import get_model_version

from mcats.ml_logic.registry import load_model, save_model
from mcats.ml_logic.registry import load_model_cnn, save_model_cnn, get_model_version_cnn

from mcats.logistic_model import train_and_classify
from mcats.wav_extraction.audio_to_df import audio_to_df

from mcats.ml_logic.params import (
    LOCAL_DATA_PATH,
    LOCAL_SONG_PATH
)

from mcats.data_sources.local_disk import save_local_raw_csv
from mcats.data_sources.local_disk import split_data_csv

from mcats.json_extraction.extract_json_data import extract_json_data
from mcats.ml_logic.preprocessor_cnn import preprocess_cnn
from mcats.wav_extraction.file_to_prediction_cnn import normalize_volume, file_to_mfcc,predict_song_cat
from mcats.ml_logic.registry import load_preprocessor


def import_wav(wav_dir: str = None):
    """
    Creating a csv dataset by extracting features from music wav files
    """

    print("\n⭐️ Use case: Extracting wav files features into a csv dataset")

    if wav_dir is None:

        wav_dir = os.path.join(
            os.path.expanduser(LOCAL_DATA_PATH),
            "wav_files",
            DATASET_SIZE)

        audio_df = audio_to_df(wav_dir, False)
        save_local_raw_csv(DATASET_SIZE, audio_df)
        return None

    else:
        audio_df = audio_to_df(wav_dir, True)
        return audio_df


def split_data():
    split_data_csv(DATASET_SIZE)
    return None


def preprocess(source_type = 'train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ Use case: preprocess")

    # Iterate on the dataset, in chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"

    while (True):
        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(
            source_name=source_name,
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # Break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_chunk = data_chunk_cleaned.drop("genre", axis=1)
        y_chunk = data_chunk_cleaned[["genre"]]

        if source_type == "train":
            X_processed_chunk = preprocess_features(X_chunk, False, False)
        elif source_type == "val":
           X_processed_chunk = preprocess_features(X_chunk, True, False)

        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1)
        )

        # Save and append the chunk
        is_first = chunk_id == 0

        save_chunk(
            destination_name=destination_name,
            is_first=is_first,
            data=data_processed_chunk
        )

        chunk_id += 1

    if row_count == 0:
        print("\n✅ No new data for the preprocessing 👌")
        return None

    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def train():
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load a validation set common to all chunks, used to early stop model training
    data_val_processed = get_chunk(
        # Remember to create a val_processed_{VALIDATION_DATASET_SIZE} table on the google bigquery cloud
        source_name=f"val_processed_{VALIDATION_DATASET_SIZE}",
        index=0,  # retrieve from first row
        chunk_size=None
    )  # Retrieve all further data

    if data_val_processed is None:
        print("\n✅ no data to train")
        return None

    data_val_processed = data_val_processed.to_numpy()

    X_val_processed = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]

    model = None
    model = load_model()  # production model


    # Model params for svm
    kernel='rbf'
    C=3

    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)
        # Remember to create a train_processed_{DATASET_SIZE} table on the google bigquery cloud
        data_processed_chunk = get_chunk(
            source_name=f"train_processed_{DATASET_SIZE}",
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Check whether data source contain more data
        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        data_processed_chunk = data_processed_chunk.to_numpy()

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        # Increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        # INITIALIZE SVM MODEL
        if model is None:
            model = initialize_model(kernel, C)


        # Train SVM
        model, accuracy = train_model(model,
                            X_train_chunk,
                            y_train_chunk
        )

        metrics_val_chunk = accuracy
        metrics_val_list.append(metrics_val_chunk)
        print(f"Chunk accuracy: {round(metrics_val_chunk,2)}")



        # Check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    print(f"\n✅ trained on {row_count} rows")

    params = dict(

        #Model parameters for svm
        kernel = kernel,
        C=C,

        # Package behavior
        context="train",
        chunk_size=CHUNK_SIZE,

        # Data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,
        model_version=get_model_version()
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(accuracy=accuracy))

    return accuracy



def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    number_to_genre = {0:'hiphop',
                     1:'classical',
                     2:'blues',
                     3:'metal',
                     4:'jazz',
                     5:'country',
                     6:'pop',
                     7:'rock',
                     8:'disco',
                     9:'reggae'}

    pred_folder = os.path.join(
            os.path.expanduser(LOCAL_DATA_PATH),
            "wav_files",
            "user_input")

    X_pred = import_wav(pred_folder)
    X_processed = preprocess_features(X_pred, False, True)
    model = load_model()
    y_pred = model.predict(X_processed)[0]

    print("\n✅ The genre of the song is: ", number_to_genre[y_pred])

    return y_pred


def pred_CNN(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    encoder = load_preprocessor()
    model = load_model_cnn()

    number_to_genre = {'hiphop': 0,
                        'classical': 1,
                        'pop': 2,
                        'electronic': 3,
                        'metal': 4,
                        'rock': 5,
                        'country': 6,
                        'reggae': 7}

    pred_folder = os.path.join(
            os.path.expanduser(LOCAL_DATA_PATH),
            "wav_files",
            "user_input")

    for filename in os.listdir(pred_folder):
        file_path = os.path.join(pred_folder, filename)
        y_pred =  predict_song_cat(file_path,model,encoder)
        print(f"\n✅ The genre of the song {filename} is {y_pred}")

    return y_pred





def train_CNN(X,y):

    # Model params
    learning_rate = 0.0001
    batch_size = 32
    patience = 10

    model = None
    model = load_model_cnn()  # production model

    # Initialize model
    if model is None:
        model = initialize_model(X)

    # (Re-)compile and train the model
    model = compile_model(model, learning_rate)
    model, history = train_model(
        model,
        X,
        y,
        batch_size=batch_size,
        patience=patience
    )

    accuracy = np.min(history.history['accuracy'])

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",

        # Data source
        val_set_size=VALIDATION_DATASET_SIZE,
        model_version=get_model_version_cnn()
    )

    # Save model
    save_model_cnn(model=model, params=params, metrics=dict(accuracy=accuracy))

    return accuracy



if __name__ == '__main__':

    try:
        print('Let\'s go the app is thinking !!')


        # Transform  the audio input into usable data
        # import_wav()
        # Split raw data between training set and validation set
        # split_data()
        # Preprocess raw training data
        # preprocess()
        # Preprocess raw validation data
        # preprocess(source_type='val')
        # Train the preprocessed data
        # train()
        # Score the model
        #evaluate()
        # Classify the given song
        # pred()

        #path = os.path.join(
        #    os.path.expanduser(LOCAL_DATA_PATH),
        #    "JSON",
        #    "Jamendo")

        #X, y = extract_json_data(path)
        #X,y = preprocess_cnn(X,y)
        #train_CNN(X,y)

        pred_CNN()

    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

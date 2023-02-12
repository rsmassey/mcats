import numpy as np
import pandas as pd
import os

from colorama import Fore, Style

from mcats.ml_logic.data import clean_data, get_chunk, save_chunk
from mcats.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from mcats.ml_logic.params import CHUNK_SIZE, DATASET_SIZE, VALIDATION_DATASET_SIZE
from mcats.ml_logic.preprocessor import preprocess_features
from mcats.ml_logic.utils import get_dataset_timestamp
from mcats.ml_logic.registry import get_model_version

from mcats.ml_logic.registry import load_model, save_model


from logistic_model import train_and_classify
import audio_to_df
import feature_extraction
from file_to_prediction import file_to_prediction

from ml_logic.params import (
    LOCAL_DATA_PATH,
    LOCAL_SONG_PATH
)


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

        X_chunk = data_chunk_cleaned.drop("label", axis=1)
        y_chunk = data_chunk_cleaned[["label"]]

        X_processed_chunk = preprocess_features(X_chunk)

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

    # Model params
    learning_rate = 0.001
    batch_size = 256
    patience = 2

    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

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

        # Initialize model
        if model is None:
            model = initialize_model(X_train_chunk)

        # (Re-)compile and train the model incrementally
        model = compile_model(model, learning_rate)
        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_processed, y_val)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(f"Chunk MAE: {round(metrics_val_chunk,2)}")

        # Check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    print(f"\n✅ trained on {row_count} rows with MAE: {round(val_mae, 2)}")

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",
        chunk_size=CHUNK_SIZE,

        # Data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,
        model_version=get_model_version(),
        dataset_timestamp=get_dataset_timestamp(),
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(mae=val_mae))

    return val_mae


def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: evaluate")

    # Load new data
    new_data = get_chunk(
        source_name=f"val_processed_{DATASET_SIZE}",
        index=0,
        chunk_size=None
    )  # Retrieve all further data

    if new_data is None:
        print("\n✅ No data to evaluate")
        return None

    new_data = new_data.to_numpy()

    X_new = new_data[:, :-1]
    y_new = new_data[:, -1]

    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    # Save evaluation
    params = dict(
        dataset_timestamp=get_dataset_timestamp(),
        model_version=get_model_version(),

        # Package behavior
        context="evaluate",

        # Data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=len(X_new)
    )

    save_model(params=params, metrics=dict(mae=mae))

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    from taxifare.ml_logic.registry import load_model

    if X_pred is None:

        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]
        ))

    model = load_model()
    X_pred["pickup_datetime"] = pd.to_datetime(X_pred["pickup_datetime"])

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':

    #X is rock which is 7
    #X = [0.35286835736042205,4.209018291530816,-4.139893281373414,-0.36151849976613537,-0.324048278541866,-0.0220056891316001,3.212987597216296,-0.06988163698286191,3.687788756573567,-0.8068050832828481,2.09099896879914,0.12455206251666981,2.8137793128025876,0.8445060683875784,1.0404135772669734,-1.0342002325426003,0.8374341972447673,1.2213124090210183,0.269077953711084,-1.2811136245806218,2.1692740843216702,0.958075004977099,0.7884862891323723,-1.6959285821925387,0.2038282950946665,-0.13479373421641466,0.5722168754367878,-0.991672033163992,0.2573896177949083,0.12379604003811286,0.3561963557079899,-1.6057797354532017,0.3981013689321286,-0.10590154820221166,-0.3247620464197768,-1.5495749389009055,0.02751838896081336,0.5544470708956186,-0.21232181042115794,-0.834468436244732,-0.25262895316696476,-0.2682830529352942,-0.4531911543703119,-1.4978249384100708,0.16634324899234657,-0.9228654629196373,-0.04444391263880561,-0.6813035323056931,-0.3021834503055795,0.16608864320890349,-0.2448159703721963,-0.32982132671195785,-0.49226571630971333,-0.3918646889383616,0.018836399049240594,0.18458706090893628,-0.2439589039123528,-0.6858964019059938,-0.45167158337697055,-1.134074747775978,-0.40968465821791306,-0.26531265226949685,-0.8709209127561559,1.1915202977521062,-0.18383049748419786,2.0226116286297247,0.2952246055329907,1.13531058199389,-0.28508273004512624,-0.22881120524536436,-0.34092160243209746,0.15607086977922408,0.010026071633334025,0.8940469090158178,0.17761881500814497,2.773594997582999,0.1985018549490116,3.484481876208549,0.35047900603311644,2.6054772906919785,0.2773489981734091,0.6058823969171808,0.3639214827512356,-0.041324527387303994,0.2548780611263805,-0.04199178413029917,-0.28951039315162425,0.9006024012779197,-0.3590705198955154]

    # X = file_to_prediction(LOCAL_SONG_PATH)

    try:
        print('Let\'s go the app is thinking !!')

        # preprocess()
        # preprocess(source_type='val')
        # train()
        # pred()
        # evaluate()

        #Transform  the audio input into usable data

        #Preprocess the raw data


        #train a model and use the provided input to classify
        result = train_and_classify(X)
        print(f'The genre of the provided song is: {result}')

    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

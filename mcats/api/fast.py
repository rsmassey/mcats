import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

from colorama import Fore, Style
from mcats.ml_logic.preprocessor_cnn import preprocess_cnn
from mcats.wav_extraction.file_to_prediction_cnn import normalize_volume, file_to_mfcc,predict_song_cat
from mcats.ml_logic.registry import load_preprocessor
from mcats.ml_logic.registry import load_model_cnn, save_model_cnn, get_model_version_cnn

from mcats.ml_logic.params import (
    LOCAL_DATA_PATH,
    LOCAL_SONG_PATH
)


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )


@app.post("/classify")
def predict(song_name: str):

    pred_file = os.path.join(
            os.path.expanduser(LOCAL_DATA_PATH),
            "wav_files",
            "user_input", song_name)

    SAK = os.environ.get("SAK")


    credentials = service_account.Credentials.from_service_account_file(
        f'{SAK}'
    )
    client = storage.Client(credentials=credentials)

    # Get the bucket you want to upload the file to
    bucket = client.get_bucket('mcats_bucket_1')

    # Download a file to the bucket
    blob = bucket.blob(f'user_input/{song_name}')
    blob.download_to_filename(pred_file)

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

    y_pred =  predict_song_cat(pred_file,model,encoder)
    print(f"\n✅ The genre of the song {song_name} is {y_pred}")

    return y_pred


@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}

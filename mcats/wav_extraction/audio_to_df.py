from mcats.wav_extraction.feature_extraction import normalize_volume
from mcats.wav_extraction.feature_extraction import extract_features
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler




def audio_to_df(audio_files_root_dir: str, pred: bool):

    # Create a dataframe with the names of each feature
    columns = ['tempo', 'beats_mean', 'beats_var',
    'zero_crossings_mean', 'zero_crossings_var',
    'spectral_centroids_mean', 'spectral_centroids_var', 'spectral_rolloff_mean',
    'spectral_rolloff_var']
    for i in range(40):
        columns.extend((f'mfcc_{i+1}_mean', f'mfcc_{i+1}_var'))
    columns.append('genre')

    df = pd.DataFrame(columns=columns)

    # Data for classification of one song
    # Code to modify if we want to implement the classification of several songs
    if pred:
        if os.path.isdir(audio_files_root_dir):
            for filename in os.listdir(audio_files_root_dir):
                file_path = os.path.join(audio_files_root_dir, filename)
                print(f'Processing file: {filename}')
                X_norm, sr = normalize_volume(file_path)
                X_clip = X_norm[30 * sr: 30 * sr *2]
                X_clip_features = extract_features(X_norm, sr)
                #X_clip_features = extract_features(np.array(X_clip), sr)
                #X_clip_features = np.array(X_clip_features).T
                #X_clip_features = np.array(X_clip_features).reshape(1, 90)
                df = X_clip_features

    # Dataset to feed the model for modeling and training the algorithm
    else:
        genres = []

        # Loop through the folders of each genre
        for genre_dir in os.listdir(audio_files_root_dir):
            print(f'Processing {genre_dir} folder')
            if genre_dir not in genres:
                genres.append(genre_dir)
            genre_path = os.path.join(audio_files_root_dir, genre_dir)
            if os.path.isdir(genre_path):
                # Loop through the audio files in the folder
                for filename in os.listdir(genre_path):
                    print(f'Processing file: {filename}')
                    file_path = os.path.join(genre_path, filename)
                    # Check if the file is a wav file
                    if file_path.endswith('.wav'):
                        # Extract the features from the audio file, avoiding any corrupt files
                        try:
                            y_norm, sr = normalize_volume(file_path)
                            features = extract_features(y_norm, sr)
                            features.append(genre_dir)
                            # Add a new row to the dataframe with the index being the name of the audio file
                            df.loc[filename] = features
                            print(f'File {filename} processed')
                        except Exception as e:
                            # Print the error message and move on to the next file
                            print(f"Error processing file {file_path}: {e}")
                            continue
        # Change genre names into a number that can be used in for training a model
        genre_to_number = {genre: number for number, genre in enumerate(genres)}
        df['genre'] = df['genre'].apply(lambda x: genre_to_number[x])

    return df

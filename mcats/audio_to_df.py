from feature_extraction import normalize_volume
from feature_extraction import extract_features
import os
import pandas as pd

def audio_to_df(audio_files_root_dir):
    # Create a dataframe with the names of each feature
    columns = ['tempo', 'beats', 'zero_crossings_mean', 'zero_crossings_var',
    'spectral_centroids_mean', 'spectral_centroids_var', 'spectral_rolloff_mean',
    'spectral_rolloff_var']
    for i in range(40):
        columns.extend((f'mfcc_{i+1}_mean', f'mfcc_{i+1}_var'))
    df = pd.DataFrame(columns=columns)

    # Loop through the folders
    for genre_dir in os.listdir(audio_files_root_dir):
        genre_path = os.path.join(audio_files_root_dir, genre_dir)
        if os.path.isdir(genre_path):
            # Loop through the audio files in the folder
            for filename in os.listdir(genre_path):
                file_path = os.path.join(genre_path, filename)
                # Extract the features from the audio file
                y_norm, sr = normalize_volume(file_path)
                features = extract_features(y_norm, sr)
                # Add a new row to the dataframe
                df.loc[len(df)] = features

    return df

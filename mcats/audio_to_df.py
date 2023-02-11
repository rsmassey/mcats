from feature_extraction import normalize_volume
from feature_extraction import extract_features
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def audio_to_df(audio_files_root_dir):
    # Create a dataframe with the names of each feature

    columns = ['tempo', 'beats_mean', 'beats_var',
    'zero_crossings_mean', 'zero_crossings_var',
    'spectral_centroids_mean', 'spectral_centroids_var', 'spectral_rolloff_mean',
    'spectral_rolloff_var']
    for i in range(40):
        columns.extend((f'mfcc_{i+1}_mean', f'mfcc_{i+1}_var'))
    columns.append('genre')

    df = pd.DataFrame(columns=columns)

    genres = []

    # Loop through the folders of each genre
    for genre_dir in os.listdir(audio_files_root_dir):
        if genre_dir not in genres:
            genres.append(genre_dir)
        genre_path = os.path.join(audio_files_root_dir, genre_dir)
        if os.path.isdir(genre_path):
            # Loop through the audio files in the folder
            for filename in os.listdir(genre_path):
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
                    except Exception as e:
                        # Print the error message and move on to the next file
                        print(f"Error processing file {file_path}: {e}")
                        continue
    # Change genre names into a number that can be used in for training a model
    genre_to_number = {genre: number for number, genre in enumerate(genres)}
    df['genre'] = df['genre'].apply(lambda x: genre_to_number[x])

    return df, genre_to_number

def df_to_scaled_df(df):
    # Use standard scaler from sklearn library
    scaler = StandardScalar()

    # Remove the genre column and standardize the features
    df_without_genre = df.iloc[:, :-1].copy()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_without_genre),
                               columns=df_without_genre.columns, index=df_without_genre.index)

    # Re-add the genre column to the dataframe
    df_standardized_with_genre = pd.concat([df_standardized, df['genre']], axis=1, ignore_index=False)

    return df_standardized_with_genre

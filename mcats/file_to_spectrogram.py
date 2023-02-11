import librosa
from feature_extraction import normalize_volume

def file_to_db_spectrogram(file_path)

    # Load file into time series and sample rate
    y_norm, sr = normalize_volume(file_path)

    # Perform short time Fourier transform
    y_stft = librosa.stft(y_norm)
    # Convert the STFT into decibal (absolute value because negative values give an error for logarithms)
    y_db = librosa.amplitude_to_db(abs(y_stft))

    return y_db, sr

def show_db_spectrogram(y_db, sr)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(y_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def folders_to_spectrograms(audio_files_root_dir):

    genres = []
    spectrograms_genre = []

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
                        y_db, sr = file_to_db_spectrogram(file_path)

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

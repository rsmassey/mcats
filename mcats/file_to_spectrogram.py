import librosa

def file_to_db_spectrogram(file_path)

    # Load file into time series and sample rate
    y, sr = librosa.load(file_path)

    # Perform short time Fourier transform
    y_stft = librosa.stft(y)
    # Convert the STFT into decibal (absolute value because negative values give an error for logarithms)
    y_db = librosa.amplitude_to_db(abs(y_stft))

    return y_db, sr

def show_db_spectrogram(y_db, sr)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(y_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

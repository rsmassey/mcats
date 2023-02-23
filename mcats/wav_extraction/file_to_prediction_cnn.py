import librosa
import numpy as np

def normalize_volume(file_path):
    audio, sr = librosa.load(file_path, offset=30.0, duration=30.0)
    audio_norm = librosa.util.normalize(audio, axis=0)
    return audio_norm

def file_to_mfcc(file_path, n_seg, i):

    hop_length = 512 # num. of samples
    n_fft = 2048 # num. of samples for window
    n_mfcc = 13 # num. of mfccs

    sr = 22050
    samples_per_audio = sr * 30
    samples_per_segment = int(samples_per_audio / n_seg)

    start = samples_per_segment * i
    end = start + samples_per_segment

    # Load file into time series and sample rate
    audio_norm = normalize_volume(file_path)

    # Convert the STFT into decibal (absolute value because negative values give an error for logarithms)
    mfcc = librosa.feature.mfcc(y=audio_norm[start:end], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    return mfcc

def predict_song_cat(file_path, model, encoder):

    n_seg = 10

    segment_mfccs = []
    predictions = np.zeros(8)

    for i in range(n_seg):
        segment_mfcc = file_to_mfcc(file_path, n_seg, i)
        target_shape = (13, 130)
        pad_width = [(0, max(0, target_shape[i] - segment_mfcc.shape[i])) for i in range(len(target_shape))]
        # Pad the array with zeros
        padded_mfcc = np.pad(segment_mfcc, pad_width=pad_width, mode='constant', constant_values=0)
        padded_mfcc = np.array(padded_mfcc)
        padded_mfcc = padded_mfcc[np.newaxis,...,np.newaxis]
        segment_mfccs.append(padded_mfcc)

        prediction = np.ravel(model.predict(padded_mfcc))
        predictions += prediction

    predictions_int = np.round(predictions).astype(int)
    pred = encoder.inverse_transform(predictions_int.reshape(1,-1))

    return pred[0][0]

import matplotlib.pyplot as plt
import streamlit as st
import librosa
import keras
import numpy as np
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

"""
# Welcome to Streamlit!
Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
In the meantime, below is an example of what you can do with just a few lines of code:
"""

def normalize_volume(music_file):
    audio, sr = librosa.load(music_file, offset=30.0, duration=30.0)
    audio_norm = librosa.util.normalize(audio, axis=0)
    return audio_norm

def file_to_mfcc(audio_norm, n_seg, i):

    hop_length = 512 # num. of samples
    n_fft = 2048 # num. of samples for window
    n_mfcc = 13 # num. of mfccs

    sr = 22050
    samples_per_audio = sr * 30
    samples_per_segment = int(samples_per_audio / n_seg)

    start = samples_per_segment * i
    end = start + samples_per_segment

    # Convert the STFT into decibal (absolute value because negative values give an error for logarithms)
    mfcc = librosa.feature.mfcc(y=audio_norm[start:end], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    return mfcc

music_file = st.file_uploader("Choose a music file")

if music_file is not None:
    audio_norm = normalize_volume(music_file)
    audio_stft = librosa.stft(audio_norm)
    audio_db = librosa.amplitude_to_db(abs(audio_stft))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(audio_db, sr=22050, x_axis='time', y_axis='hz')
    plt.colorbar()
    st.pyplot()
    st.write("Here is the spectrogram!")

model = keras.models.load_model('cnn2.h5')
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

if encoder is not None:
    segment_mfccs = []
    predictions = np.zeros(8)
    target_shape = (13, 130)
    for i in range(10):
        segment_mfcc = file_to_mfcc(audio_norm, 10, i)
        pad_width = [(0, max(0, target_shape[i] - segment_mfcc.shape[i])) for i in range(len(target_shape))]
        padded_mfcc = np.pad(segment_mfcc, pad_width=pad_width, mode='constant', constant_values=0)
        padded_mfcc = np.array(padded_mfcc)
        padded_mfcc = padded_mfcc[np.newaxis,...,np.newaxis]
        segment_mfccs.append(padded_mfcc)
        prediction = np.ravel(model.predict(padded_mfcc))
        predictions += prediction
    predictions_int = np.round(predictions).astype(int)
    pred = predictions_int.reshape(1,-1)
    pred = encoder.inverse_transform(predictions_int.reshape(1,-1))
    genre = pred[0][0]

    st.write(f"The genre of this song is ...")
    st.markdown(f"<h1 style='text-align: center; color: red;'>{genre}</h1>", unsafe_allow_html=True)

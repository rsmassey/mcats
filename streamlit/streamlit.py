import matplotlib.pyplot as plt
import streamlit as st
import librosa
import keras
import os
import numpy as np
import pickle
import base64
import IPython.display as ipd
import time

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

col1, col2 = st.columns([1, 1])

with col1:

    st.title('MCATs: Music Category Analyzing Tools')

with col2:

    st.header('Brought to you by:')
    st.markdown(f'<h1 style="font-size: 32px;">Alexandre Bun, Ryan Massey, Sarah Deutchman, Steven Tin</h1>', unsafe_allow_html=True)

def normalize_volume(music_file):
    audio, sr = librosa.load(music_file, offset=30.0, duration=30.0)
    st.audio(audio, sample_rate=sr)
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

    # Load file into time series and sample rate
    # audio_norm = normalize_volume(music_file)

    # Convert the STFT into decibal (absolute value because negative values give an error for logarithms)
    mfcc = librosa.feature.mfcc(y=audio_norm[start:end], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    return mfcc

def run_prediction(audio_norm, model):
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

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=audio_norm)
    beats_mean = beats.mean()
    
    st.markdown(f"<h2 style='text-align: left;'> The tempo is {tempo:.1f} beats per minute.</h2>", unsafe_allow_html=True)
        
    # Frequencies
    pitches, magnitudes = librosa.piptrack(y=audio_norm, fmin=20)
    min_freq = pitches[pitches != 0].min()
    max_freq = pitches.max()

    st.markdown(f"<h2 style='text-align: left;'>The frequencies range from {min_freq:.1f} Hz to {max_freq:.1f} Hz.</h2>", unsafe_allow_html=True)
    time.sleep(3)
    st.markdown(f"<h1 style='text-align: left;'>The genre of this song is ...</h1>", unsafe_allow_html=True)

    file_ = open(f'/app/mcats/streamlit/{genre}_2.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" width="750">',
        unsafe_allow_html=True,
    )

col1, col2 = st.columns([1, 1])

with col1:
    music_file = None
    music_file = st.file_uploader("Choose a music file")
    
    if music_file is not None:
        audio_norm = normalize_volume(music_file)
        audio_stft = librosa.stft(audio_norm)
        audio_db = librosa.amplitude_to_db(abs(audio_stft))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(audio_db, sr=22050, x_axis='time', y_axis='log')
        plt.colorbar()
        st.pyplot()

    else:
        file_ = open('/app/mcats/streamlit/record.gif', 'rb')
        contents = file_.read()
        data_url = base64.b64encode(contents).decode('utf-8')
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" style="display: flex; justify-content: center;">',
            unsafe_allow_html=True,
        )

with col2:
    st.markdown("""---""") 
    time.sleep(3)
    model = keras.models.load_model('cnn2.h5')
    with open('/app/mcats/streamlit/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    try:
        result = run_prediction(audio_norm, model)
    except:
        pass

    add_bg_from_local('/app/mcats/streamlit/background_image.png')

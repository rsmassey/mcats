import librosa, librosa.display
import os
import pandas as pd
import numpy as np

def normalize_volume(file_path):
    y, sr = librosa.load(file_path)
    y_norm = librosa.util.normalize(y, axis=0)
    return y_norm, sr

def extract_features(y_norm, sr):

    features = []

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y_norm, sr=sr)
    beats_mean = beats.mean()
    beats_var = beats.var()
    features.extend((tempo, beats_mean, beats_var))

    # Zero crossings
    zero_crossings = librosa.zero_crossings(y=y_norm, pad=False)
    zero_crossings_mean = zero_crossings.mean()
    zero_crossings_var = zero_crossings.var()
    features.extend((zero_crossings_mean, zero_crossings_var))

    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y_norm, sr=sr)[0]
    spectral_centroids_mean = spectral_centroids.mean()
    spectral_centroids_var = spectral_centroids.var()
    features.extend((spectral_centroids_mean,spectral_centroids_var))

    # Specral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_norm, sr=sr)[0]
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_rolloff_var = spectral_rolloff.var()
    features.extend((spectral_rolloff_mean, spectral_rolloff_var))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=40)
    for mfcc in mfccs:
        features.append(mfcc.mean())
        features.append(mfcc.var())

    return features

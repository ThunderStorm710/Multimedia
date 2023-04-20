#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa
# https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd
# https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis



def lerFicheiroCsv(fich: str):
    if not fich:
        return None
    # ler o arquivo top100_features.csv e criar um array numpy
    features = np.genfromtxt(fich, delimiter=',')
    print(features)
    return features


def normalizarFeatures(features):
    # normalizar as features no intervalo [0, 1]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    print(normalized_features)
    np.savetxt("Features Extraidas.txt", normalized_features)
    return normalized_features


def extrairFeatures():
    features_list = []

    #lista = ["Q1", "Q2", "Q3", "Q4"]
    lista = ["Q1"]
    i = 0
    for dir in lista:
        for filename in os.listdir(f"MER_audio_taffc_dataset/{dir}"):
            if i == 5:
                break
            print(f"FICHEIRO --> {filename}")
            if filename.endswith('.mp3'):
                audio_file = os.path.join(f"MER_audio_taffc_dataset/{dir}", filename)

                # Extrair as features espectrais
                y, sr = librosa.load(audio_file, sr=22050, mono=True)

                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                spectral_flatness = librosa.feature.spectral_flatness(y=y)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

                # Extrair as features temporais
                f0, voiced_flags, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                rms = librosa.feature.rms(y=y)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

                # Extrair outras features
                tempo = librosa.beat.tempo(y=y, sr=sr)

                # Adicionar as features a uma lista
                features = np.concatenate((mfcc.flatten(), spectral_centroid.flatten(),
                                           spectral_bandwidth.flatten(), spectral_contrast.flatten(),
                                           spectral_flatness.flatten(), spectral_rolloff.flatten(), f0.flatten(),
                                           rms.flatten(), zero_crossing_rate.flatten(),
                                           tempo.flatten()))
                features_list.append(features)
                i += 1

    print(features_list)
    features_array = np.vstack(features_list)
    np.savetxt("Features Array.txt", features_array)

    return features_array


def calcularEstatisticas(dados):
    listaDados = []
    for i in dados:

        mean_features = np.mean(i, axis=0)
        std_features = np.std(i, axis=0)
        skewness_features = skew(i, axis=0)
        kurtosis_features = kurtosis(i, axis=0)
        median_features = np.median(i, axis=0)
        max_features = np.max(i, axis=0)
        min_features = np.min(i, axis=0)

    # Criar um array numpy 2D com as estatísticas descritivas
        stats_features = np.vstack(
        [mean_features, std_features, skewness_features, kurtosis_features, median_features, max_features,
         min_features])

    # Normalizar as features no intervalo [0, 1]
        norm_features = (stats_features - np.min(stats_features, axis=1, keepdims=True)) / (
                    np.max(stats_features, axis=1, keepdims=True) - np.min(stats_features, axis=1, keepdims=True))

        listaDados.append(norm_features)

    # Salvar as features normalizadas num arquivo numpy
    np.save('stats_features_librosa.npy', listaDados)


if __name__ == "__main__":
    plt.close('all')

    # --- Load file
    # fName = "--/Queries/MT0000414517.mp3"
    fName = "Queries/MT0000202045.mp3"
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono=mono)
    print(y.shape)
    print(fs)

    # --- Play Sound
    sd.play(y, sr, blocking=False)

    features = lerFicheiroCsv('Features - Audio MER/top100_features.csv')
    featuresNormalizadas = normalizarFeatures(features)
    extrairFeatures()

    # --- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)

    # --- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # --- Extract features
    rms = librosa.feature.rms(y=y)
    rms = rms[0, :]
    print(rms.shape)
    times = librosa.times_like(rms)
    plt.figure(), plt.plot(times, rms)
    plt.xlabel('Time (s)')
    plt.title('RMS')

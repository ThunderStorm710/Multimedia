import librosa
import librosa.display
import librosa.beat
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis


def lerFicheiroCsv(fich: str):
    if not fich:
        return None
    np.set_printoptions(suppress=True)
    features = np.genfromtxt(fich, delimiter=',', dtype=float, skip_header=1)
    features = features[:, 1:len(features[1]) - 1]
    print(features)
    return features


def normalizarFeatures(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    print(normalized_features)
    np.savetxt("Features Extraidas.csv", normalized_features)
    return normalized_features


def extrairFeatures():
    features_list = []

    i = 0
    for filename in os.listdir(f"MER_audio_taffc_dataset/Songs"):
        if i == 5:
            break
        print(f"FICHEIRO --> {filename}")
        if filename.endswith('.mp3'):
            ficheiro = os.path.join(f"MER_audio_taffc_dataset/Songs", filename)
            y, sr = librosa.load(ficheiro, sr=22050, mono=True)

            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc = mfcc[:13, :]
            print("MFCC = ", mfcc.shape)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            print("spectral_centroid = ", spectral_centroid.shape)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            print("spectral_bandwidth = ", spectral_bandwidth.shape)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            print("spectral_contrast = ", spectral_contrast.shape)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            print("spectral_flatness = ", spectral_flatness.shape)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            print("spectral_rolloff = ", spectral_rolloff.shape)

            # Extrair as features temporais
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            print("f0 = ", f0.shape)

            rms = librosa.feature.rms(y=y)
            print("rms = ", rms.shape)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
            print("zero_crossing_rate = ", zero_crossing_rate.shape)

            # Extrair outras features
            tempo = librosa.beat.tempo(y=y, sr=sr)
            print("tempo = ", tempo.shape)

            # Adicionar as features a uma lista

            features_list.append([mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness,
                                  spectral_rolloff, f0, rms, zero_crossing_rate, tempo])
            i += 1

    return features_list


def calcularEstatisticas(dados):
    print(dados[0])
    listaDados = []
    for i in dados:
        if type(i) == list:
            for j in i:
                mean_features = np.mean(j, axis=0)
                std_features = np.std(j, axis=0)
                skewness_features = skew(j, axis=0)
                kurtosis_features = kurtosis(j, axis=0)
                median_features = np.median(j, axis=0)
                max_features = np.max(j, axis=0)
                min_features = np.min(j, axis=0)

                listaDados.append(
                    [mean_features, std_features, skewness_features, kurtosis_features, median_features, max_features,
                     min_features])
        else:
            mean_features = np.mean(i, axis=0)
            std_features = np.std(i, axis=0)
            skewness_features = skew(i, axis=0)
            kurtosis_features = kurtosis(i, axis=0)
            median_features = np.median(i, axis=0)
            max_features = np.max(i, axis=0)
            min_features = np.min(i, axis=0)

            listaDados.append(
                [mean_features, std_features, skewness_features, kurtosis_features, median_features, max_features,
                 min_features])



        # Normalizar as features no intervalo [0, 1]
        #norm_features = (stats_features - np.min(stats_features, axis=1, keepdims=True)) / (np.max(stats_features, axis=1, keepdims=True) - np.min(stats_features, axis=1, keepdims=True))

        #listaDados.append(norm_features)
    np.savetxt("Estatisitcas.txt", listaDados)
    listaDados = np.array(listaDados)

    # Salvar as features normalizadas num arquivo numpy
    np.save('stats_features_librosa.csv', listaDados)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.sqrt(np.sum(x1 ** 2))
    norm_x2 = np.sqrt(np.sum(x2 ** 2))
    return dot_product / (norm_x1 * norm_x2)


if __name__ == "__main__":
    plt.close('all')

    # --- Load file
    fName = "Queries/MT0000202045.mp3"
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    # y, fs = librosa.load(fName, sr=sr, mono=mono)
    # print(y.shape)
    # print(fs)

    # --- Play Sound
    # sd.play(y, sr, blocking=False)
    features = lerFicheiroCsv('Features - Audio MER/top100_features.csv')
    featuresNormalizadas = normalizarFeatures(features)
    features = extrairFeatures()
    calcularEstatisticas(features)

    # --- Plot sound waveform
    # plt.figure()
    # librosa.display.waveshow(y)

    # --- Plot spectrogram
    # Y = np.abs(librosa.stft(y))
    # Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    # ax.set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # --- Extract features
    # rms = librosa.feature.rms(y=y)
    # rms = rms[0, :]
    # print(rms.shape)
    # times = librosa.times_like(rms)
    # plt.figure(), plt.plot(times, rms)
    # plt.xlabel('Time (s)')
    # plt.title('RMS')

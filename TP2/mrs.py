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
    info = np.genfromtxt(fich, delimiter=',', dtype=float, skip_header=1)
    info = info[:, 1:len(info[1]) - 1]
    return info


def normalizarFeatures(info):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(info)
    np.savetxt("Features Top 100.csv", normalized_features)
    return normalized_features


def extrairFeatures():
    #features_list = np.empty(10,)
    features_list = []
    i = 0

    if not "stats_features_librosa.npy" in os.listdir():
        features_list = np.load("stats_features_librosa.npy", allow_pickle=True)
        print("TIPO", type(features_list.astype(list)))
        print("TIPO", features_list)
        return features_list.astype(list)
    else:
        for filename in os.listdir(f"MER_audio_taffc_dataset/Songs"):
            if i == 10:
                break
            print(f"FICHEIRO --> {filename}")

            if filename.endswith('.mp3'):
                y, sr = librosa.load("MER_audio_taffc_dataset/Songs/" + filename, sr=22050, mono=True)

                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc = mfcc[:13, :]
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                spectral_flatness = librosa.feature.spectral_flatness(y=y)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                f0 = f0.reshape(1, f0.shape[0])
                rms = librosa.feature.rms(y=y)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
                tempo = librosa.beat.tempo(y=y, sr=sr)
                '''
                a = np.array([mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, f0, rms, zero_crossing_rate, tempo])
                features_list = np.insert(features_list,  [a], axis=0)
                print(features_list.shape)
                print(len(features_list[0]))
                for l in features_list[0]:
                    print(l)
                    '''

                features_list.append([mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, f0, rms, zero_crossing_rate, tempo])
                i += 1
        print(len(features_list), len(features_list[0][0]))
        #print(features_list.shape)
        return features_list


def calcularEstatisticas(dados):
    #listaDados = np.empty(shape=(0,))
    listaDados = []

    for i in dados:  # MUSICAS
        listaAux = []
        contador = 0
        aux = 0
        for j in i:  # CADA FEATURE

            shape = j.shape
            print(shape, "_--")
            if len(j) > 1:
                for k in j:
                    mean = np.mean(k)
                    std = np.std(k)
                    skewness = skew(k)
                    curtose = kurtosis(k)
                    median = np.median(k)
                    valorMax = np.max(k)
                    valorMin = np.min(k)


                    listaAux.extend([mean, std, skewness, curtose, median, valorMax, valorMin])
                    contador += 7
                    aux += 1
            else:
                mean = np.mean(j)
                std = np.std(j)
                skewness = skew(j)
                curtose = kurtosis(j)
                median = np.median(j)
                valorMax = np.max(j)
                valorMin = np.min(j)

                listaAux.extend([mean, std, skewness, curtose, median, valorMax, valorMin])
                contador += 7
                aux += 1

        #listaDados = np.append(listaDados, listaAux, axis=0)
        listaDados.append(listaAux)

    print("DADOS = ", len(listaDados), len(listaDados[0]))
    np.save('stats_features_librosa', listaDados)
    print("NORMALIZAR")
    normalizar(listaDados)
    print("-----------------------------------")
    #np.savetxt("Estatisticas.txt", listaDados)

    return listaDados


def obterMusicas():
    listaMusicas = np.empty(shape=(0,))
    for ficheiro in os.listdir(f"MER_audio_taffc_dataset/Songs"):
        if ficheiro.endswith('.mp3'):
            y, sr = librosa.load("MER_audio_taffc_dataset/Songs/" + ficheiro, sr=22050, mono=True)
            print(len(y))
            listaMusicas = np.append(listaMusicas,y, axis=0)

    print("FIM = ", len(listaMusicas))
    return listaMusicas


def normalizar(lista):
    array = np.array(lista)
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(len(lista)):
        print(i, " VALOR")
        aux = array[:, i]  # usa a sintaxe do NumPy para acessar a coluna do array
        print(aux)
        aux = scaler.fit_transform(aux.reshape(-1, 1))  # redimensiona a coluna para ter formato adequado
        array[:, i] = aux.flatten()  # atualiza a coluna normalizada no array
        print(aux)

    print(len(array))
    np.save("ArrayNormalizado", array)
    return array


def obterDistancias():
    lista = obterMusicas()
    distanciaEuclidiana = []
    distanciaManhattan = []
    distanciaCosseno = []
    source = 22050
    for i in range(len(lista)):
        spectral_centroid1 = librosa.feature.spectral_centroid(y=lista[i], sr=source)
        for j in range(len(lista)):
            spectral_centroid2 = librosa.feature.spectral_centroid(y=lista[j], sr=source)
            der = euclidean_distance(spectral_centroid1[0], spectral_centroid2[0])
            dmr = manhattan_distance(spectral_centroid1[0], spectral_centroid2[0])
            dcr = cosine_similarity(spectral_centroid1[0], spectral_centroid2[0])
            distanciaEuclidiana.append(der)
            distanciaManhattan.append(dmr)
            distanciaCosseno.append(dcr)
    print(distanciaEuclidiana)
    np.save(distanciaEuclidiana)
    np.save(distanciaManhattan)
    np.save(distanciaCosseno)


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
    fName = "Queries/MT0000202045.mp3"
    s = 22050
    mono = True
    warnings.filterwarnings("ignore")

    # --- Play Sound
    # sd.play(y, sr, blocking=False)
    features = lerFicheiroCsv('Features - Audio MER/top100_features.csv')
    featuresNormalizadas = normalizarFeatures(features)

    features = extrairFeatures()
    stats = calcularEstatisticas(features)
    normalizarFeatures(stats)

    obterDistancias()

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

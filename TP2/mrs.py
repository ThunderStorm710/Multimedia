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
from scipy.spatial.distance import cosine


def lerFicheiroCsv(fich: str, string):
    if not fich:
        return None
    np.set_printoptions(suppress=True)
    if string:
        info = np.genfromtxt(fich, delimiter=',', dtype=str, skip_header=1)
    else:
        info = np.genfromtxt(fich, delimiter=',', dtype=float, skip_header=1)

    info = info[:, 1:len(info[1]) - 1]
    return info


def normalizarFeatures(info):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(info)
    np.savetxt("Features Top 100.csv", normalized_features)
    return normalized_features


def extrairFeatures():
    features_list = []
    i = 0

    if "Features - 900x190.npy" in os.listdir():
        features_list = np.load("Features - 900x190.npy", allow_pickle=True)
        return features_list
    else:
        for filename in os.listdir(f"MER_audio_taffc_dataset/Songs"):
            '''
            if i == 20:
                break
            '''
            print(f"FICHEIRO --> {filename}")

            if filename.endswith('.mp3'):
                y, sr = librosa.load("MER_audio_taffc_dataset/Songs/" + filename, sr=22050, mono=True)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                lista = calcularStats(mfcc)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                lista.extend(calcularStats(spectral_centroid))
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                lista.extend(calcularStats(spectral_bandwidth))
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                lista.extend(calcularStats(spectral_contrast))
                spectral_flatness = librosa.feature.spectral_flatness(y=y)
                lista.extend(calcularStats(spectral_flatness))
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                lista.extend(calcularStats(spectral_rolloff))
                f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                f0 = f0.reshape(1, f0.shape[0])
                lista.extend(calcularStats(f0))
                rms = librosa.feature.rms(y=y)
                lista.extend(calcularStats(rms))
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
                lista.extend(calcularStats(zero_crossing_rate))
                tempo = librosa.beat.tempo(y=y, sr=sr)
                lista.extend(tempo)

                features_list.append(lista)
                i += 1
        lista = np.array(features_list)
        np.save("Features - 900x190", lista)
        np.savetxt("Features - 900x190.csv", features_list, delimiter=",")
        return features_list


def calcularStats(dados):
    listaAux = []
    for j in dados:  # CADA FEATURE
        mean = np.mean(j)
        std = np.std(j)
        skewness = skew(j)
        curtose = kurtosis(j)
        median = np.median(j)
        valorMax = np.max(j)
        valorMin = np.min(j)

        listaAux.extend([mean, std, skewness, curtose, median, valorMax, valorMin])

    return listaAux


def obterNomesMusicas():
    listaMusicas = []
    for ficheiro in os.listdir(f"MER_audio_taffc_dataset/Songs"):
        if ficheiro.endswith('.mp3'):
            listaMusicas.append(ficheiro)

    return listaMusicas


def normalizar(lista):
    array = np.array(lista)
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(len(lista[0])):
        aux = array[:, i]  # usa a sintaxe do NumPy para acessar a coluna do array
        aux = scaler.fit_transform(aux.reshape(-1, 1))  # redimensiona a coluna para ter formato adequado
        array[:, i] = aux.flatten()  # atualiza a coluna normalizada no array

    np.save("Features Normalizadas 900x190", array)
    np.savetxt("Features Normalizadas - 900x190.csv", array, delimiter=",")

    return array


def obterDistancias(lista, featuresTop):
    ficheiros = os.listdir()
    if "der.npy" in ficheiros and "derTop100.npy" in ficheiros:
        distanciaEuclidiana = np.load("der.npy", allow_pickle=True)
        distanciaEuclidianaTop100 = np.load("derTop100.npy", allow_pickle=True)
        return distanciaEuclidiana, distanciaEuclidianaTop100

    else:

        distanciaEuclidiana = np.zeros((900, 900))
        distanciaManhattan = np.zeros((900, 900))
        distanciaCosseno = np.zeros((900, 900))
        distanciaEuclidianaTop100 = np.zeros((900, 900))
        distanciaManhattanTop100 = np.zeros((900, 900))
        distanciaCossenoTop100 = np.zeros((900, 900))

        for i in range(len(lista)):
            for j in range(i, len(lista)):
                if i == j:
                    distanciaEuclidiana[i][j] = -1
                    distanciaManhattan[i][j] = -1
                    distanciaCosseno[i][j] = -1
                else:

                    aux1 = np.array(lista[i])
                    aux2 = np.array(lista[j])
                    distanciaEuclidiana[i][j] = distanciaEuclidiana[j][i] = np.linalg.norm(aux1 - aux2)
                    distanciaManhattan[i][j] = distanciaManhattan[j][i] = np.sum(np.abs(aux1 - aux2))
                    distanciaCosseno[i][j] = distanciaCosseno[j][i] = cosine(aux1, aux2)

        for i in range(len(featuresTop)):
            for j in range(i, len(featuresTop)):
                if i == j:
                    distanciaEuclidianaTop100[i][j] = -1
                    distanciaManhattanTop100[i][j] = -1
                    distanciaCossenoTop100[i][j] = -1
                else:
                    aux1 = np.array(featuresTop[i])
                    aux2 = np.array(featuresTop[j])
                    distanciaEuclidianaTop100[i][j] = distanciaEuclidianaTop100[j][i] = np.linalg.norm(aux1 - aux2)
                    distanciaManhattanTop100[i][j] = distanciaManhattanTop100[j][i] = np.sum(np.abs(aux1 - aux2))
                    distanciaCossenoTop100[i][j] = distanciaCossenoTop100[j][i] = cosine(aux1, aux2)

        np.save("der", distanciaEuclidiana)
        np.save("dmr", distanciaManhattan)
        np.save("dcr", distanciaCosseno)

        np.savetxt("der.csv", distanciaEuclidiana, delimiter=",")
        np.savetxt("dmr.csv", distanciaManhattan, delimiter=",")
        np.savetxt("dcr.csv", distanciaCosseno, delimiter=",")

        np.save("derTop100", distanciaEuclidianaTop100)
        np.save("dmrTop100", distanciaManhattanTop100)
        np.save("dcrTop100", distanciaCossenoTop100)

        np.savetxt("derTop100.csv", distanciaEuclidianaTop100, delimiter=",")
        np.savetxt("dmrTop100.csv", distanciaManhattanTop100, delimiter=",")
        np.savetxt("dcrTop100.csv", distanciaCossenoTop100, delimiter=",")

        return distanciaEuclidiana, distanciaEuclidianaTop100


def rankingSimilaridade(info, infoTop100):
    diretoria = os.listdir("Queries/")
    musicas = obterNomesMusicas()
    for ficheiro in diretoria:
        if ficheiro.endswith(".mp3"):
            listaMusicas = []
            indice = musicas.index(ficheiro)  # Indice da musica/query
            linha = info[indice]  # Distancia entre a query e todas as musicas
            array = np.array(linha)
            array = np.argsort(array)[0:20]

            for i in array:
                listaMusicas.append(musicas[i])
            print(f"Query {ficheiro} --> Features Extraidas {listaMusicas}")

            listaMusicas = []
            linha = infoTop100[indice]  # Distancia entre a query e todas as musicas
            array = np.array(linha)
            array = np.argsort(array)[0:20]
            for i in array:
                listaMusicas.append(musicas[i])
            print(f"Query {ficheiro} --> Metadados {listaMusicas}")

        else:
            print(f"Query {ficheiro} não encontrada...")


def correspondenciaMetadados():
    ficheiros = os.listdir(f"MER_audio_taffc_dataset")
    listaSimilaridade = np.zeros((900, 900), dtype=int)
    if "panda_dataset_taffc_metadata.csv" in ficheiros:
        info = lerFicheiroCsv("MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv", True)
        for i in range(len(info)):
            moods = info[i][8].split("; ")
            moods = [c.replace('"', '') for c in moods]

            genres = info[i][10].split("; ")
            genres = [c.replace('"', '') for c in genres]
            lista = [info[i][0].replace('"', ''), info[i][2].replace('"', ''), moods, genres]

            for j in range(i, len(info)):
                similaridade = 0
                moodsJ = info[j][8].split("; ")
                moodsJ = [c.replace('"', '') for c in moodsJ]

                genresJ = info[j][10].split("; ")
                genresJ = [c.replace('"', '') for c in genresJ]

                listaJ = [info[j][0].replace('"', ''), info[j][2].replace('"', ''), moodsJ, genresJ]
                for k in range(len(lista)):
                    if k == 2 or k == 3:
                        for p in moods:
                            if p in listaJ[k]:
                                similaridade += 1
                    elif lista[k] == listaJ[k]:
                        similaridade += 1

                listaSimilaridade[i][j] = listaSimilaridade[j][i] = similaridade

    np.save("Similaridade", listaSimilaridade)
    np.savetxt("Similaridade.csv", listaSimilaridade, delimiter=",", fmt="%d")


if __name__ == "__main__":
    plt.close('all')
    fName = "Queries/MT0000202045.mp3"
    s = 22050
    mono = True
    warnings.filterwarnings("ignore")

    # --- Play Sound
    # sd.play(y, sr, blocking=False)
    featuresTop100 = lerFicheiroCsv('Features - Audio MER/top100_features.csv', False)
    featuresNormalizadasTop100 = normalizarFeatures(featuresTop100)

    features = extrairFeatures()
    # stats = calcularEstatisticas(features)
    featuresNormalizadas = normalizar(features)

    #der, derTop100 = obterDistancias(featuresNormalizadas, featuresTop100)
    #rankingSimilaridade(der, derTop100)

    correspondenciaMetadados()

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

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct


def visualizarImagem(imagem, titulo: str = None, axis: str = None):
    plt.figure()
    plt.imshow(imagem)
    plt.title(titulo)
    plt.axis(axis)
    plt.show()
    return imagem


def visualizarConjuntoImagens(comp1, comp2, comp3, titulo: list, axis: str = None, showLogaritmo: bool = False):
    if showLogaritmo:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].axis(axis)
        axs[0].imshow(np.log(abs(comp1 + 0.0001)), cmap='gray')
        axs[0].set_title(titulo[0])
        axs[1].axis(axis)
        axs[1].imshow(np.log(abs(comp2 + 0.0001)), cmap='gray')
        axs[1].set_title(titulo[1])
        axs[2].axis(axis)
        axs[2].imshow(np.log(abs(comp3 + 0.0001)), cmap='gray')
        axs[2].set_title(titulo[2])

    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].axis(axis)
        axs[0].imshow(comp1, cmap='gray')
        axs[0].set_title(titulo[0])
        axs[1].axis(axis)
        axs[1].imshow(comp2, cmap='gray')
        axs[1].set_title(titulo[1])
        axs[2].axis(axis)
        axs[2].imshow(comp3, cmap='gray')
        axs[2].set_title(titulo[2])
    plt.show()


def criarColorMap(cor: str, listaCor: list):
    if not listaCor or len(listaCor) != 2:
        return None

    cmap = clr.LinearSegmentedColormap.from_list(cor, listaCor, 256)
    data = np.outer(np.ones(100), np.linspace(-1, 1, 100))
    plt.imshow(data, cmap=cmap)
    plt.colorbar()
    plt.show()
    return cmap


def separarCanais(imagem):
    channel1 = imagem[:, :, 0]
    channel2 = imagem[:, :, 1]
    channel3 = imagem[:, :, 2]

    return channel1, channel2, channel3


def juntarCanais(channel1, channel2, channel3):
    imagem = np.zeros((channel1.shape[0], channel1.shape[1], 3))
    imagem[:, :, 0] = channel1
    imagem[:, :, 1] = channel2
    imagem[:, :, 2] = channel3
    return imagem


def pad_image(image):
    rows, cols = image.shape[:2]
    padded_rows = (32 - rows % 32) % 32
    padded_cols = (32 - cols % 32) % 32
    padded_image = np.pad(image, ((0, padded_rows), (0, padded_cols), (0, 0)), 'edge')
    return padded_image


def unpad_image(img_padded, imagem_original):
    original_shape = imagem_original.shape
    cut_rows = img_padded.shape[0] - original_shape[0]
    cut_cols = img_padded.shape[1] - original_shape[1]
    img = img_padded[0:(img_padded.shape[0] - cut_rows), 0:(img_padded.shape[1] - cut_cols)]

    return img


def aplicarColorMap(nomeFich: str):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    R, G, B = separarCanais(imagem)
    cmap = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], 256)
    imagem_colorida = cmap(R)
    plt.imshow(imagem_colorida)
    plt.show()
    cmap = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], 256)
    imagem_colorida = cmap(G)
    plt.imshow(imagem_colorida)
    plt.show()
    cmap = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], 256)
    imagem_colorida = cmap(B)
    plt.imshow(imagem_colorida)
    plt.show()


def aplicarColorMapDado(nomeFich: str, cmap):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    R, G, B = separarCanais(imagem)
    plt.imshow(R, cmap=cmap)
    plt.show()


def verYCbCr(imagem):
    Y, Cb, Cr = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(Y, cmap='gray')
    axs[0].set_title('Canal Y')
    axs[1].imshow(Cb, cmap='gray')
    axs[1].set_title('Canal Cb')
    axs[2].imshow(Cr, cmap='gray')
    axs[2].set_title('Canal Cr')
    plt.show()


def rgb_para_ycbcr(imagem):
    xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr = imagem.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr_para_rgb(imagem):
    xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    inversa = np.linalg.inv(xform)

    # xform = np.array([[1, 0, 1.402], [1, -0.344136, -.714136], [1, 1.772, 0]])
    rgb = imagem.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = np.dot(rgb, inversa.T)
    # rgb[:, :, 0] = inversa[0, 0] * imagem[:, :, 0] + inversa[0, 1] * (imagem[:, :, 1] - 128) + inversa[0, 2] * (imagem[:, :, 2] - 128)

    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    rgb = np.round(rgb)

    return rgb.astype(np.uint8)


def subamostragem(Y, Cb, Cr, downsample: str):
    Y_d = Y

    if downsample == "4:2:2":
        Cb_d = cv2.resize(Cb, (0, 0), fx=0.5, fy=1)
        Cr_d = cv2.resize(Cr, (0, 0), fx=0.5, fy=1)
    elif downsample == "4:2:0":
        Cb_d = cv2.resize(Cb, (0, 0), fx=0.5, fy=0.5)
        Cr_d = cv2.resize(Cr, (0, 0), fx=0.5, fy=0.5)
    else:
        Cb_d = Cb
        Cr_d = Cr

    return Y_d, Cb_d, Cr_d


def reconstrucao(Y, Cb_d, Cr_d):
    # Upsampling Cb e Cr para a resolução original
    Cb = cv2.resize(Cb_d, (Y.shape[1], Y.shape[0]))
    Cr = cv2.resize(Cr_d, (Y.shape[1], Y.shape[0]))

    # Não há upsampling no canal Y
    Y = Y

    return Y, Cb, Cr


def calculate_dct(channel):
    return dct(dct(channel, norm="ortho").T, norm="ortho").T


def calculate_idct(dct_channel):
    return idct(idct(dct_channel, norm="ortho").T, norm="ortho").T


def dct_bloco(imagem, BS):
    altura, largura = imagem.shape
    coefs = np.zeros((altura, largura))

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            bloco = imagem[y:y + BS, x:x + BS]
            bloco_dct = calculate_dct(bloco)
            coefs[y:y + BS, x:x + BS] = bloco_dct

    return coefs


def idct_bloco(coefs, BS):
    altura, largura = coefs.shape
    imagem = np.zeros((altura, largura))

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            bloco = coefs[y:y + BS, x:x + BS]
            bloco_idct = calculate_idct(bloco)
            imagem[y:y + BS, x:x + BS] = bloco_idct

    return imagem


def quantize_block(block, quantization_matrix):
    return np.round(np.divide(block, quantization_matrix))


def dequantize_block(block, quantization_matrix):
    return np.multiply(block, quantization_matrix).astype(float)


def dequantize_image(img, quantization_matrix):
    height, width = img.shape[:2]

    for x in range(0, height, 8):
        for y in range(0, width, 8):
            block = img[x:x + 8, y:y + 8]
            dequantized_block = np.multiply(block, quantization_matrix)
            img[x:x + 8, y:y + 8] = dequantized_block

    return img.astype(float)


def quantize_image(img, quantization_matrix):
    height, width = img.shape[:2]

    quantized_img = np.zeros((height, width))
    for x in range(0, height, 8):
        for y in range(0, width, 8):
            block = img[x:x + 8, y:y + 8]
            quantized_block = np.round(block / quantization_matrix)
            quantized_img[x:x + 8, y:y + 8] = quantized_block

    return quantized_img.astype(int)


def quantization_matrix(base_matrix, quality):
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 50 / quality
    else:
        scale = (100 - quality) / 50

    quant_matrix = np.round(scale * base_matrix)
    quant_matrix[quant_matrix > 255] = 255
    quant_matrix[quant_matrix < 1] = 1
    quant_matrix = quant_matrix.astype(np.uint8)

    return quant_matrix


def dpcm_dc(coefs, BS):
    altura, largura = coefs.shape
    difs = np.copy(coefs)
    dc_anterior = 0

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            bloco = coefs[y:y + BS, x:x + BS]
            dc = bloco[0, 0]
            dif = dc - dc_anterior
            bloco[0, 0] = dif
            difs[y:y + BS, x:x + BS] = bloco
            dc_anterior = dc

    return difs


def idpcm_dc(difs, BS):
    altura, largura = difs.shape
    coefs = np.copy(difs)
    dc_anterior = 0

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            dif = difs[y:y + BS, x:x + BS]
            dc = dif[0, 0] + dc_anterior
            dif[0, 0] = dc
            coefs[y:y + BS, x:x + BS] = dif
            dc_anterior = dc

    return coefs

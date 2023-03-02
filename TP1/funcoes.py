import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct

QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

QCbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                  [18, 21, 26, 66, 99, 99, 99, 99],
                  [24, 26, 56, 99, 99, 99, 99, 99],
                  [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99]])


def visualizarImagem(imagem, titulo: str = None, axis: str = None):
    plt.figure()
    plt.imshow(imagem)
    plt.title(titulo)
    plt.axis(axis)
    plt.show()
    return imagem


def visualizarConjuntoImagens(componente1, componente2, componente3, titulo: list, axis: str = None,
                              showLogaritmo: bool = False):
    if showLogaritmo:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].axis(axis)
        axs[0].imshow(np.log(np.abs(componente1) + 0.0001), cmap='gray')
        axs[0].set_title(titulo[0])
        axs[1].axis(axis)
        axs[1].imshow(np.log(np.abs(componente2) + 0.0001), cmap='gray')
        axs[1].set_title(titulo[1])
        axs[2].axis(axis)
        axs[2].imshow(np.log(np.abs(componente3) + 0.0001), cmap='gray')
        axs[2].set_title(titulo[2])

    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].axis(axis)
        axs[0].imshow(componente1, cmap='gray')
        axs[0].set_title(titulo[0])
        axs[1].axis(axis)
        axs[1].imshow(componente2, cmap='gray')
        axs[1].set_title(titulo[1])
        axs[2].axis(axis)
        axs[2].imshow(componente3, cmap='gray')
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
    if not channel1 or not channel2 or not channel3:
        return None

    imagem = np.zeros_like(channel1, dtype=np.uint8)
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
    return np.uint8(ycbcr)


def ycbcr_para_rgb(imagem):
    xform = np.array([[1, 0, 1.402], [1, -0.344136, -.714136], [1, 1.772, 0]])
    rgb = imagem.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.round(rgb, 0)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    rgb = rgb.astype(int)
    return np.uint8(rgb)


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
            bloco_dct = dct(dct(bloco, norm="ortho").T, norm="ortho").T
            coefs[y:y + BS, x:x + BS] = bloco_dct

    return coefs


def idct_bloco(coefs, BS):
    altura, largura = coefs.shape
    imagem = np.zeros((altura, largura))

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            bloco_coefs = coefs[y:y + BS, x:x + BS]
            bloco_idct = idct(idct(bloco_coefs, norm="ortho").T, norm="ortho").T
            imagem[y:y + BS, x:x + BS] = bloco_idct

    return imagem


def quantize_block(block, quantization_matrix):
    return np.round(np.divide(block, quantization_matrix))


def quantize_image(img, quantization_matrix):
    height, width = img.shape[:2]

    quantized_img = np.zeros((height, width))

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = img[y:y + 8, x:x + 8]
            quantized_block = quantize_block(block, quantization_matrix)
            quantized_img[y:y + 8, x:x + 8] = quantized_block
    return quantized_img


def quantization_matrix(base_matrix, quality):
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    quant_matrix = np.floor((scale * base_matrix + 50) / 100)

    quant_matrix[quant_matrix == 0] = 1

    return quant_matrix


def dpcm_dc(coefs, BS):
    altura, largura = coefs.shape
    difs = np.zeros((altura // BS, largura // BS))
    dc_anterior = 0

    for y in range(0, altura, BS):
        for x in range(0, largura, BS):
            bloco = coefs[y:y + BS, x:x + BS]
            dc = bloco[0, 0]
            dif = dc - dc_anterior
            bloco[0, 0] = dif
            difs[y // BS, x // BS] = dif
            dc_anterior = dc

    return difs, coefs


def idpcm_dc(difs, BS):
    altura, largura = difs.shape
    coefs = np.zeros((altura * BS, largura * BS))
    dc_anterior = 0

    for y in range(0, altura * BS, BS):
        for x in range(0, largura * BS, BS):
            dif = difs[y // BS, x // BS]
            dc = dif + dc_anterior
            bloco = coefs[y:y + BS, x:x + BS]
            bloco[0, 0] = dc
            dc_anterior = dc - dif

    return coefs

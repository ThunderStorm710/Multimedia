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


def dct_block(channel, bs):
    if channel.shape[0] % bs != 0 or channel.shape[1] % bs != 0:
        raise ValueError(f"Channel shape {channel.shape} is not a multiple of block size {bs}!")

    blocks = np.split(channel, channel.shape[0] // bs, axis=0)
    blocks = [np.split(block, channel.shape[1] // bs, axis=1) for block in blocks]

    dct_blocks = []
    for row in blocks:
        dct_row = []
        for block in row:
            dct_block = dct(dct(block, norm="ortho").T, norm="ortho").T
            dct_row.append(dct_block)
        dct_blocks.append(dct_row)

    return np.block(dct_blocks)


def dct2d_blocks(channel, block_size):
    rows, cols = channel.shape

    # Divide a imagem em blocos não sobrepostos
    block_rows = rows // block_size
    block_cols = cols // block_size
    blocks = np.zeros((block_rows, block_cols, block_size, block_size))

    for i in range(block_rows):
        for j in range(block_cols):
            blocks[i, j, :, :] = channel[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

    # Aplica a DCT em cada bloco
    dct_blocks = np.zeros((block_rows, block_cols, block_size, block_size))
    for i in range(block_rows):
        for j in range(block_cols):
            dct_blocks[i, j, :, :] = dct(dct(blocks[i, j, :, :], norm='ortho').T, norm='ortho').T

    # Concatena os resultados em uma única matriz
    dct_full = np.zeros((rows, cols))
    for i in range(block_rows):
        for j in range(block_cols):
            dct_full[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = dct_blocks[i, j, :, :]

    return dct_full


def idct_block(dct_coeffs, bs):
    """
    Applies IDCT to a channel using blocks of size bs x bs.

    Arguments:
    dct_coeffs -- 2D numpy array representing the DCT coefficients
    bs -- block size

    Returns:
    2D numpy array with channel values
    """
    # Check if the DCT coefficients have the right shape
    if dct_coeffs.shape[0] % bs != 0 or dct_coeffs.shape[1] % bs != 0:
        raise ValueError(f"DCT coefficients shape {dct_coeffs.shape} is not a multiple of block size {bs}!")

    # Split the DCT coefficients into blocks
    blocks = np.split(dct_coeffs, dct_coeffs.shape[0] // bs, axis=0)
    blocks = [np.split(block, dct_coeffs.shape[1] // bs, axis=1) for block in blocks]

    # Apply IDCT to each block
    idct_blocks = []
    for row in blocks:
        idct_row = []
        for block in row:
            idct_block = idct(idct(block.T, norm="ortho").T, norm="ortho")
            idct_row.append(idct_block)
        idct_blocks.append(idct_row)

    # Convert the list of blocks back into a 2D numpy array
    return np.block(idct_blocks)


def quantize_block(block, quantization_matrix):
    return np.round(np.divide(block, quantization_matrix))


def quantize_image(img, quantization_matrix):
    """
    Quantiza todos os blocos 8x8 de uma imagem utilizando uma matriz de quantização.

    Args:
        img: array numpy 2D com a imagem.
        quantization_matrix: array numpy 2D de shape (8, 8) com a matriz de quantização.

    Returns:
        Um novo array numpy 2D com a imagem quantizada.
    """
    # Calcula o tamanho da imagem
    height, width = img.shape[:2]

    # Cria um array vazio para armazenar a imagem quantizada
    quantized_img = np.zeros((height, width))

    # Loop que percorre a imagem em blocos de tamanho 8x8
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            # Seleciona um bloco 8x8 da imagem
            block = img[y:y + 8, x:x + 8]

            # Quantiza o bloco utilizando a matriz de quantização
            quantized_block = quantize_block(block, quantization_matrix)

            # Armazena o bloco quantizado na imagem final
            quantized_img[y:y + 8, x:x + 8] = quantized_block

    # Retorna a imagem quantizada
    return quantized_img


def quantize_dct(dct_block, quality_factor=50):
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]])
    if quality_factor < 1:
        quality_factor = 1
    if quality_factor > 99:
        quality_factor = 99
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor
    quantization_matrix = ((scale * quantization_table) + 50) // 100
    quantization_matrix[quantization_matrix == 0] = 1
    quantized_dct_block = np.round(dct_block / quantization_matrix)
    return quantized_dct_block, quantization_matrix


def quantization_matrix(quality):
    """
    Retorna uma matriz de quantização para um determinado fator de qualidade.

    Args:
        quality: inteiro que representa o fator de qualidade desejado (entre 1 e 100).

    Returns:
        Um array numpy 2D com a matriz de quantização.
    """
    # Define a matriz de quantização padrão
    base_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

    # Verifica se o fator de qualidade é válido
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    # Calcula o fator de escala
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    # Multiplica a matriz de quantização padrão pelo fator de escala
    quant_matrix = np.floor((scale * base_matrix + 50) / 100)

    # Verifica se algum elemento da matriz de quantização é zero
    # e ajusta para evitar divisão por zero
    quant_matrix[quant_matrix == 0] = 1

    # Retorna a matriz de quantização
    return quant_matrix

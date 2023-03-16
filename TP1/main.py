from funcoes import *

blockSize = 8
qualidade = 75

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

matrizY = quantization_matrix(QY, qualidade)
matrizCbCr = quantization_matrix(QCbCr, qualidade)


def encode(nomeFich: str):
    global qualidade

    if not nomeFich:
        return None
    image = plt.imread(nomeFich)
    # visualizarImagem(image, nomeFich, "off")
    '''----------------------------------------------- EX 3 ---------------------------------------------------------'''
    # aplicarColorMap(nomeFich)
    # colormap = criarColorMap("purple-ish", [(0, 0, 0), (0.6, 0.1, 0.9)])
    # aplicarColorMapDado(nomeFich, colormap)
    '''----------------------------------------------- EX 4 ---------------------------------------------------------'''
    padded_image = pad_image(image)
    # visualizarImagem(padded_image, "PADDED IMAGE", "off")
    '''----------------------------------------------- EX 5 ---------------------------------------------------------'''
    ycbcr_image = rgb_para_ycbcr(padded_image)
    # visualizarImagem(ycbcr_image, "RGB para YCbCr", "off")
    # verYCbCr(ycbcr_image)
    '''----------------------------------------------- EX 6 ---------------------------------------------------------'''
    Y, Cb, Cr = separarCanais(ycbcr_image)
    Y_d, Cb_d, Cr_d = subamostragem(Y, Cb, Cr, "4:2:2")
    # visualizarConjuntoImagens(Y_d, Cb_d, Cr_d, ["Y_d", "Cb_d", "Cr_d"], 'off', True)
    '''----------------------------------------------- EX 7 ---------------------------------------------------------'''
    # Y_dct = calculate_dct(Y_d)
    # Cb_dct = calculate_dct(Cb_d)
    # Cr_dct = calculate_dct(Cr_d)

    # visualizarConjuntoImagens(Y_dct, Cb_dct, Cr_dct, ["Y_dct", "Cb_dct", "Cr_dct"], 'off', True)

    Y_dct8 = dct_bloco(Y_d, blockSize)
    Cb_dct8 = dct_bloco(Cb_d, blockSize)
    Cr_dct8 = dct_bloco(Cr_d, blockSize)

    #visualizarConjuntoImagens(Y_dct8, Cb_dct8, Cr_dct8, ["Y_dct8", "Cb_dct8", "Cr_dct8"], 'off', True)

    # Y_dct64 = dct_bloco(Y_d, 64)
    # Cb_dct64 = dct_bloco(Cb_d, 64)
    # Cr_dct64 = dct_bloco(Cr_d, 64)

    # visualizarConjuntoImagens(Y_dct64, Cb_dct64, Cr_dct64, ["Y_dct64", "Cb_dct64", "Cr_dct64"], 'off', True)
    '''----------------------------------------------- EX 8 ---------------------------------------------------------'''
    Y_Q = quantize_image(Y_dct8, matrizY)
    Cb_Q = quantize_image(Cb_dct8, matrizCbCr)
    Cr_Q = quantize_image(Cr_dct8, matrizCbCr)

    #visualizarConjuntoImagens(Y_Q, Cb_Q, Cr_Q, ["Y_Q", "Cb_Q", "Cr_Q"], 'off', True)
    '''----------------------------------------------- EX 9 ---------------------------------------------------------'''
    difY = dpcm_dc(Y_Q, blockSize)
    difCb = dpcm_dc(Cb_Q, blockSize)
    difCr = dpcm_dc(Cr_Q, blockSize)

    #visualizarConjuntoImagens(difY, difCb, difCr, ["difY", "difCb", "difCr"], 'off', True)

    return image, difY, difCb, difCr


def decode(imagemOriginal, difY, difCb, difCr):
    global qualidade

    Y_idpcm = idpcm_dc(difY, blockSize)
    Cb_idpcm = idpcm_dc(difCb, blockSize)
    Cr_idpcm = idpcm_dc(difCr, blockSize)

    #visualizarConjuntoImagens(Y_idpcm, Cb_idpcm, Cr_idpcm, ["Y_idcpm", "Cb_idcpm", "Cr_idcpm"], 'off', True)

    Y_DQ = dequantize_image(Y_idpcm, matrizY)
    Cb_DQ = dequantize_image(Cb_idpcm, matrizCbCr)
    Cr_DQ = dequantize_image(Cr_idpcm, matrizCbCr)

    #visualizarConjuntoImagens(Y_DQ, Cb_DQ, Cr_DQ, ["Y_DQ", "Cb_DQ", "Cr_DQ"], 'off', True)

    Y_idct8 = idct_bloco(Y_DQ, blockSize)
    Cb_idct8 = idct_bloco(Cb_DQ, blockSize)
    Cr_idct8 = idct_bloco(Cr_DQ, blockSize)

    #visualizarConjuntoImagens(Y_idct8, Cb_idct8, Cr_idct8, ["Y_idct8", "Cb_idct8", "Cr_idct8"], 'off', False)

    # ex 4
    Y, Cb, Cr = reconstrucao(Y_idct8, Cb_idct8, Cr_idct8)

    #visualizarConjuntoImagens(Y, Cb, Cr, ["Y", "Cb", "Cr"], 'off', False)
    imagemYCbCr = juntarCanais(Y, Cb, Cr)
    unpadded_image = unpad_image(imagemYCbCr, imagemOriginal)
    # ex 5
    rgb_image = ycbcr_para_rgb(unpadded_image)
    #visualizarImagem(rgb_image, "IMAGEM FINAL", "off")
    # print(f"Imagem Original --> Valor do pixel [0,0]: {imagemOriginal[0][0]}")
    # print(f"Imagem Convertida --> Valor do pixel [0,0]: {rgb_image[0][0]}")
    # ex 6
    return rgb_image


if __name__ == "__main__":
    plt.close('all')
    original, componente1, componente2, componente3 = encode("imagens/barn_mountains.bmp")
    final = decode(original, componente1, componente2, componente3)
    calcularMetricasDistorcao(original, final)

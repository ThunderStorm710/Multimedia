from funcoes import *


def encode(nomeFich: str):
    if not nomeFich:
        return None
    image = plt.imread(nomeFich)
    #visualizarImagem(image, nomeFich, "off")

    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 3 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''

    aplicarColorMap(nomeFich)
    colormap = criarColorMap("purple-ish", [(0, 0, 0), (0.6, 0.1, 0.9)])
    aplicarColorMapDado(nomeFich, colormap)
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 4 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    padded_image = pad_image(image)
    #visualizarImagem(padded_image, "PADDED IMAGE", "off")
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 5 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    ycbcr_image = rgb_para_ycbcr(padded_image)
    #visualizarImagem(ycbcr_image, "RGB para YCbCr", "off")
    verYCbCr(ycbcr_image)
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 6 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    Y, Cr, Cb = separarCanais(ycbcr_image)
    Y_d, Cb_d, Cr_d = subamostragem(Y, Cb, Cr, "4:2:0")

    #cv2.imshow("Y_d", Y_d)
    #cv2.imshow("Cb_d", Cb_d)
    #cv2.imshow("Cr_d", Cr_d)
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 7 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    Y_dct = calculate_dct(Y_d)
    Cb_dct = calculate_dct(Cb_d)
    Cr_dct = calculate_dct(Cr_d)

    visualizarConjuntoImagens(Y_dct, Cb_dct, Cr_dct, ["Y_dct", "Cb_dct", "Cr_dct"], 'off', True)

    Y_idct = calculate_idct(Y_dct)
    Cb_idct = calculate_idct(Cb_dct)
    Cr_idct = calculate_idct(Cr_dct)

    visualizarConjuntoImagens(Y_idct, Cb_idct, Cr_idct, ["Y_idct", "Cb_idct", "Cr_idct"], 'off', True)

    BS = 8

    Y_dct8 = dct_bloco(Y_d, BS)
    Cb_dct8 = dct_bloco(Cb_d, BS)
    Cr_dct8 = dct_bloco(Cr_d, BS)

    Y_idct8 = idct_bloco(Y_dct8, BS)
    Cb_idct8 = idct_bloco(Cb_dct8, BS)
    Cr_idct8 = idct_bloco(Cr_dct8, BS)

    visualizarConjuntoImagens(Y_dct8, Cb_dct8, Cr_dct8, ["Y_dct8", "Cb_dct8", "Cr_dct8"], 'off', True)

    visualizarConjuntoImagens(Y_idct8, Cb_idct8, Cr_idct8, ["Y_idct8", "Cb_idct8", "Cr_idct8"], 'off', True)

    BS = 64

    Y_dct64 = dct_bloco(Y_d, BS)
    Cb_dct64 = dct_bloco(Cb_d, BS)
    Cr_dct64 = dct_bloco(Cr_d, BS)

    visualizarConjuntoImagens(Y_dct64, Cb_dct64, Cr_dct64, ["Y_dct64", "Cb_dct64", "Cr_dct64"], 'off', True)
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 8 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    matrizY = quantization_matrix(QY, 75)
    matrizCbCr = quantization_matrix(QCbCr, 75)
    quantized_img = quantize_image(Y_dct8, matrizY)

    cv2.imshow("Y_Q", quantized_img)

    quantized_img1 = quantize_image(Cb_dct8, matrizCbCr)
    cv2.imshow("Cb_Q", quantized_img1)

    quantized_img2 = quantize_image(Cr_dct8, matrizCbCr)
    cv2.imshow("Cr_Q", quantized_img2)
    '''--------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------- EX 9 ---------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------------'''
    difY, Y_dcpm = dpcm_dc(quantized_img, 8)
    cv2.imshow("Y_DCPM", Y_dcpm)
    difCb, Cb_dcpm = dpcm_dc(quantized_img1, 8)
    cv2.imshow("Cb_DCPM", Cb_dcpm)
    difCr, Cr_dcpm = dpcm_dc(quantized_img2, 8)
    cv2.imshow("Cr_DCPM", Cr_dcpm)

    Y_idcpm = decode_dc_coefficients(Y_dcpm)
    visualizarImagem(Y_idcpm, "Y_iDCPM", "off")
    #cv2.imshow("Y_iDCPM", Y_idcpm)
    #Cb_idcpm = dpcm_dc_inv(Cb_dcpm)
    #cv2.imshow("Cb_iDCPM", Cb_idcpm)
    #Cr_idcpm = dpcm_dc_inv(Cr_dcpm)
    #cv2.imshow("Cr_iDCPM", Cr_idcpm)

    return image, padded_image, ycbcr_image, Y_d, Cb_d, Cr_d


def decode(imagemOriginal, imagemPadding, imagemYCbCr, Y_d, Cb_d, Cr_d):
    # ex 4
    unpadded_image = unpad_image(imagemPadding, imagemOriginal)
    #visualizarImagem(unpadded_image, "PADDING REMOVED", "off")
    # ex 5
    rgb_image = ycbcr_para_rgb(imagemYCbCr)
    #visualizarImagem(rgb_image, "YCbCr para RGB", "off")
    print(f"Imagem Original --> Valor do pixel [0,0]: {imagemOriginal[0][0]}")
    print(f"Imagem Convertida --> Valor do pixel [0,0]: {rgb_image[0][0]}")
    # ex 6
    Y_u, Cb_u, Cr_u = reconstrucao(Y_d, Cb_d, Cr_d)
    #cv2.imshow("Y_u", Y_u)
    #cv2.imshow("Cb_u", Cb_u)
    #cv2.imshow("Cr_u", Cr_u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    plt.close('all')
    original, padded, ycbcr, Y_d, Cb_d, Cr_d = encode("imagens/barn_mountains.bmp")
    decode(original, padded, ycbcr, Y_d, Cb_d, Cr_d)

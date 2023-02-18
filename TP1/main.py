import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2


def encode(nomeFich: str):
    if not nomeFich:
        return None
    image = plt.imread(nomeFich)
    visualizarImagem(image, nomeFich, "off")
    # ex 3
    aplicarColorMap(nomeFich)
    colormap = criarColorMap("purple-ish", [(0, 0, 0), (0.6, 0.1, 0.9)])
    aplicarColorMapDado(nomeFich, colormap)
    # ex 4
    padded_image = pad_image(image)
    visualizarImagem(padded_image, "PADDED IMAGE", "off")
    # ex 5
    ycbcr_image = rgb_para_ycbcr(padded_image)
    visualizarImagem(ycbcr_image, "RGB para YCbCr", "off")
    verYCbCr(ycbcr_image)
    # ex 6
    Y, Cr, Cb = separarCanais(ycbcr_image)
    Y_d, Cb_d, Cr_d = subamostragem(Y, Cb, Cr, "4:2:2")

    cv2.imshow("Y_d", Y_d)
    cv2.imshow("Cb_d", Cb_d)
    cv2.imshow("Cr_d", Cr_d)
    return image, padded_image, ycbcr_image, Y_d, Cb_d, Cr_d


def decode(imagemOriginal, imagemPadding, imagemYCbCr, Y_d, Cb_d, Cr_d):
    # ex 4
    unpadded_image = unpad_image(imagemPadding, imagemOriginal)
    visualizarImagem(unpadded_image, "PADDING REMOVED", "off")
    # ex 5
    rgb_image = ycbcr_para_rgb(imagemYCbCr)
    visualizarImagem(rgb_image, "YCbCr para RGB", "off")
    print(f"Imagem Original --> Valor do pixel [0,0]: {imagemOriginal[0][0]}")
    print(f"Imagem Convertida --> Valor do pixel [0,0]: {rgb_image[0][0]}")
    # ex 6
    Y_u, Cb_u, Cr_u = reconstrucao(Y_d, Cb_d, Cr_d)
    cv2.imshow("Y_u", Y_u)
    cv2.imshow("Cb_u", Cb_u)
    cv2.imshow("Cr_u", Cr_u)
    img_reconstruida = cv2.merge((Y_u, Cr_u, Cb_u))

    # Exibe a imagem reconstruída
    cv2.imshow('Imagem reconstruída', img_reconstruida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualizarImagem(imagem, titulo: str = None, axis: str = None):
    plt.figure()
    plt.imshow(imagem)
    plt.title(titulo)
    plt.axis(axis)
    plt.show()
    return imagem


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


if __name__ == "__main__":
    plt.close('all')
    original, padded, ycbcr, Y_d, Cb_d, Cr_d = encode("imagens/barn_mountains.bmp")
    decode(original, padded, ycbcr, Y_d, Cb_d, Cr_d)


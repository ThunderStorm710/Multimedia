import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def visualizarImagem(nomeFich: str):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    plt.figure()
    plt.imshow(imagem)
    plt.title(nomeFich)
    plt.axis("on")
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


def verYCbCr(imagem):
    Y, Cb, Cr = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(Y, cmap='gray')
    axs[0].set_title('Canal Y')
    axs[1].imshow(Cb, cmap='jet')
    axs[1].set_title('Canal Cb')
    axs[2].imshow(Cr, cmap='coolwarm')
    axs[2].set_title('Canal Cr')
    plt.show()


def aplicarColorMapDado(nomeFich: str, cmap):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    R, G, B = separarCanais(imagem)
    imagem_colorida = cmap(R)
    # imagem_colorida = juntarRGB(cmap(R), cmap(G), cmap(B))
    plt.imshow(imagem_colorida)
    plt.show()


def rgb_para_ycbcr(imagem):
    R, G, B = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    ycbcr_image = np.dstack((Y, Cb, Cr))
    return ycbcr_image


def ycbcr_para_rgb(imagem):
    Y, Cb, Cr = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb_image = np.dstack((R, G, B))
    return rgb_image


if __name__ == "__main__":
    plt.close("all")
    im = visualizarImagem("imagens/barn_mountains.bmp")
    '''
    criarColorMap("gray", [(0, 0, 0), (1, 1, 1)])
    criarColorMap("red", [(0, 0, 0), (1, 0, 0)])
    criarColorMap("green", [(0, 0, 0), (0, 1, 0)])
    criarColorMap("blue", [(0, 0, 0), (0, 0, 1)])
    aplicarColorMap("imagens/peppers.bmp")
    '''
    pad = pad_image(im)
    i = unpad_image(pad, im)
    plt.title("PADDED")
    plt.imshow(pad)
    plt.show()
    plt.title("UNPADDED")
    plt.imshow(i)
    plt.show()

    # colormap = criarColorMap("purple-ish", [(0, 0, 0), (0.6, 0.1, 0.9)])  #
    # aplicarColorMapDado("imagens/peppers.bmp", colormap)  #
    # plt.imshow(pad)
    # plt.show()
    # ex 5
    ycbcr_image = rgb_para_ycbcr(im)
    plt.imshow(ycbcr_image)
    plt.title("RGB para YCbCr")
    plt.show()

    plt.imshow(ycbcr_image[:, :, 0])
    plt.title("RGB para YCbCr")
    plt.show()
    plt.imshow(ycbcr_image[:, :, 1])
    plt.title("RGB para YCbCr")
    plt.show()
    plt.imshow(ycbcr_image[:, :, 2])
    plt.title("RGB para YCbCr")
    plt.show()
    verYCbCr(ycbcr_image)
    rgb_image = ycbcr_para_rgb(ycbcr_image)
    plt.imshow(rgb_image)
    plt.title("YCbCr para RGB")
    plt.show()

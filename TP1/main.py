import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def visualizarImagem(nomeFich: str):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    plt.figure()
    plt.imshow(imagem)
    plt.title("IMAGEM")
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


def separarRGB(imagem):
    red = imagem[:, :, 0]
    green = imagem[:, :, 1]
    blue = imagem[:, :, 2]

    return red, green, blue


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


def juntarRGB(red, green, blue):
    if not red or not green or not blue:
        return None

    imagem = np.zeros_like(red, dtype=np.uint8)
    imagem[:, :, 0] = red
    imagem[:, :, 1] = green
    imagem[:, :, 2] = blue
    return imagem


def aplicarColorMap(nomeFich: str):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    R, G, B = separarRGB(imagem)
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
    R, G, B = separarRGB(imagem)
    imagem_colorida = cmap(R)
    # imagem_colorida = juntarRGB(cmap(R), cmap(G), cmap(B))
    plt.imshow(imagem_colorida)
    plt.show()


def rgb_conversion_ycbcr(imagem):
    '''
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = imagem.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
    '''
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr_conversion_rgb(imagem):
    '''
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = imagem.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))
    '''
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


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
    '''
    colormap = criarColorMap("purple-ish", [(0, 0, 0), (0.6, 0.1, 0.9)])  #
    aplicarColorMapDado("imagens/peppers.bmp", colormap)  #
    # plt.imshow(pad)
    # plt.show()
    # ex 5
    imagem = plt.imread("imagens/barn_mountains.bmp")
    ycbcr_image = rgb_conversion_ycbcr(imagem)
    plt.imshow(ycbcr_image)
    plt.show()
    rgb_image = ycbcr_conversion_rgb(ycbcr_image)
    plt.imshow(rgb_image)
    plt.show()
    '''

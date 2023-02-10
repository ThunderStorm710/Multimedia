import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def visualizarImagem(nomeFich: str):
    if not nomeFich:
        return None

    imagem = plt.imread(nomeFich)
    plt.figure()
    plt.imshow(imagem)
    plt.axis("off")
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


def separarRGB(imagem):
    red = imagem[:, :, 0]
    green = imagem[:, :, 1]
    blue = imagem[:, :, 2]

    return red, green, blue


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


if __name__ == "__main__":
    plt.close("all")
    visualizarImagem("imagens/barn_mountains.bmp")
    criarColorMap("gray", [(0, 0, 0), (1, 1, 1)])
    criarColorMap("red", [(0, 0, 0), (1, 0, 0)])
    criarColorMap("green", [(0, 0, 0), (0, 1, 0)])
    criarColorMap("blue", [(0, 0, 0), (0, 0, 1)])
    aplicarColorMap("imagens/peppers.bmp")

import librosa
import librosa.display
import librosa.beat
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine
from numpy.lib.stride_tricks import as_strided


def frame(x: np.ndarray, frame_length: int, hop_length: int, axis: int = -1, writeable: bool = False,
          subok: bool = False) -> np.ndarray:
    x = np.array(x, copy=False, subok=subok)

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def extrairRMS():
    # Ler arquivo MP3
    print("------------------------")
    for filename in os.listdir(f"MER_audio_taffc_dataset/Songs"):
        y, sr = librosa.load("MER_audio_taffc_dataset/Songs/" + filename, sr=22050, mono=True)

        x = frame(y, 93, 23)
        print(x.shape)
        print(y, y.shape)
        rms = np.sqrt(np.mean(np.square(y)))
        print(rms.shape, rms)
        rms = librosa.feature.rms(y=y)
        print(rms.shape, rms)
        return rms
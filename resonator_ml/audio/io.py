import numpy as np
from scipy.io import wavfile



def load_wav_mono(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float32)

    # stereo → mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # normalisieren
    x /= np.max(np.abs(x)) + 1e-12
    return fs, x
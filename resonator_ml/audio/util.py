import numpy as np
import librosa
def frame_batch_generator(x:np.ndarray, batch_size:int, drop_last:bool=False):
    """
    x: NumPy-Array beliebiger Dimensionen, z. B. (N,), (N, C), (N, C, F), ...
    batch_size: Anzahl der Slices pro Batch
    drop_last: falls True -> vollständige Batches erzwingen

    Gibt Batches der Form (batch_size, ...) oder (rest_size, ...) aus.
    """
    n = x.shape[0]
    for i in range(0, n, batch_size):
        batch = x[i:i+batch_size]
        if drop_last and batch.shape[0] < batch_size:
            return
        yield batch


def _stft_mag(x, sr, n_fft=2048, hop_length=None, win_length=None):
    if hop_length is None:
        hop_length = int(0.05 * sr) # ~50 ms
    hop_length = min(hop_length, n_fft)
    if win_length is None:
        win_length = hop_length
    win_length = min(win_length, n_fft)

    S = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )
    return np.abs(S)

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-8)
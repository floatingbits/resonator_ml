import numpy as np

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




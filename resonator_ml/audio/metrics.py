from typing import Protocol
import numpy as np
from resonator_ml.audio.util import _stft_mag
from librosa.feature import melspectrogram

class Metric(Protocol):
    def compute(self, reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:...

class LogStftDistanceMetric(Metric):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute(self, ref, est, sr) -> float:
        s_ref = _stft_mag(ref, sr)
        s_est = _stft_mag(est, sr)

        log_ref = np.log(s_ref + self.eps)
        log_est = np.log(s_est + self.eps)

        return np.mean((log_ref - log_est) ** 2)

class SpectraConvergenceMetric(Metric):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute(self, ref, est, sr) -> float:
        s_ref = _stft_mag(ref, sr)
        s_est = _stft_mag(est, sr)

        numerator = np.linalg.norm(s_ref - s_est, ord='fro')
        denominator = np.linalg.norm(s_ref, ord='fro') + self.eps

        return numerator / denominator

class MelSpectrogrammDistanceMetric(Metric):
    def __init__(self, eps=1e-8, n_mels=64):
        self.eps = eps
        self.n_mels = n_mels

    def compute(self, ref, est, sr) -> float:
        hop_length = int(0.05 * sr)

        M_ref = melspectrogram(
            y=ref,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=hop_length
        )
        M_est = melspectrogram(
            y=est,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=hop_length
        )

        log_ref = np.log(M_ref + self.eps)
        log_est = np.log(M_est + self.eps)

        return np.mean((log_ref - log_est) ** 2)

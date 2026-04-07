from resonator_ml.audio.metrics import LogStftDistanceMetric, SpectraConvergenceMetric, MelSpectrogrammDistanceMetric


def audio_perceptual_metrics():
    return {
        "log_stft": LogStftDistanceMetric(),
        "spectral_convergence": SpectraConvergenceMetric(),
        "mel_distance": MelSpectrogrammDistanceMetric(),
    }
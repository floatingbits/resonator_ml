from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainingParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    loss_function: Callable
    seq_length: int
    energy_lambda: float
    spectral_lambda: float
    phase_lambda: float
    correlation_lambda: float
    correlation_loss_threshold: float
    max_training_data_frames: int = 80000
    use_energy_and_decay: bool = False
    energy_window_size_in_s: float = 0.01
    decay_time_measure_time_in_s: float = 0.05
    training_data_glob: str = "[0-9].wav"

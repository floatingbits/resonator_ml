from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainingParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    loss_function: Callable
    max_training_data_frames: int = 80000

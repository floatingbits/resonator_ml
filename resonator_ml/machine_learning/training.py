from dataclasses import dataclass
from typing import Callable

import torch

from resonator_ml.machine_learning.custom_loss_functions import relative_l1, log_magnitude_mse


@dataclass
class TrainingParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    loss_function: Callable


class TrainingParameterFactory:
    def create_parameters(self, version: str):
        match version:
            case "v1.1":
                return TrainingParameters(batch_size=2000, epochs=2000, learning_rate= 1e-4, loss_function=torch.nn.MSELoss())
            case "v1.2":
                return TrainingParameters(batch_size=32, epochs=200, learning_rate=1e-4,
                                          loss_function=torch.nn.MSELoss())
            case "v2":
                return TrainingParameters(batch_size=20000, epochs=100, learning_rate=1e-4,
                                          loss_function=relative_l1)
            case "v2.1":
                return TrainingParameters(batch_size=32, epochs=200, learning_rate=1e-4,
                                          loss_function=relative_l1)
            case "v2.2":
                return TrainingParameters(batch_size=32, epochs=200, learning_rate=1e-4,
                                          loss_function=log_magnitude_mse)
            case "v2.3":
                return TrainingParameters(batch_size=32, epochs=100, learning_rate=1e-4,
                                          loss_function=relative_l1)
            case "dev":
                return TrainingParameters(batch_size=32, epochs=5, learning_rate=1e-4,
                                          loss_function=relative_l1)
            case _:
                return TrainingParameters(batch_size=20000, epochs=200, learning_rate=1e-4, loss_function=torch.nn.MSELoss())

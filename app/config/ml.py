from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    batch_size: int = 32
    learning_rate: float = 1e-3
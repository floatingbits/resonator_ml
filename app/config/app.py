from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str = "data/results"
    decay_path: str = "decay_only"
    instrument_name: str = "Strat_E"
    resonator_version = "v1"
    training_parameters_version: str = "dev"
    generation_index = "4"
    sample_rate: int = 44100
    base_frequency: float = 82.46


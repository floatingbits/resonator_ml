from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    results_path: str = "data/results"
    resonator_results_sub_path: str = "resonator/workspace"
    resonator_training_path: str = "data/processed/decay_only"
    cache_path: str = "data/cache"
    loop_filer_training_data_cache_sub_path: str = "loop_filter_training_data"
    instrument_name: str = "Strat_E"
    resonator_version = "v3_1"
    training_parameters_version: str = "v2.4"
    model_generation_index = "1"
    sound_generation_index = "1"
    sample_rate: int = 44100
    base_frequency: float = 82.46



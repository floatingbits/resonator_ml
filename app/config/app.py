from dataclasses import dataclass, field
from typing import Optional

from resonator_ml.machine_learning.custom_loss_functions import relative_l1_with_penalty
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkParameters, DelayPattern
from resonator_ml.machine_learning.training.parameters import TrainingParameters


@dataclass()
class Config:
    model_file_path: Optional[str] = None
    initialize_sound_file_path: Optional[str] = None
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
    base_frequency: float = 82.97
    output_soundfile_length: float = 5
    reuse_last_model_file: bool = False
    training_parameters: TrainingParameters = field(default_factory=lambda : TrainingParameters(batch_size=128, epochs=500, learning_rate=3e-4,
                                          loss_function=relative_l1_with_penalty, max_training_data_frames=600000, use_energy_and_decay=True))
    neural_network_parameters: NeuralNetworkParameters = field(default_factory=lambda : NeuralNetworkParameters(
        num_hidden_per_layer=1500,
        num_hidden_layers=1,
        delay_patterns=[DelayPattern(1,10, 10), DelayPattern(0.75, 10, 10), DelayPattern(0.25, 10, 10), DelayPattern(0.5, 10, 10)],
        use_decay_feature=True
    ))




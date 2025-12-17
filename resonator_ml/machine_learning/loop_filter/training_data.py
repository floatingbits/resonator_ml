from typing import Callable

import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass
from resonator_ml.machine_learning.custom_loss_functions import relative_l1, log_magnitude_mse

from resonator_ml.audio.util import frame_batch_generator
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonatorFactory, NeuralNetworkDataset


class FilepathGenerator:
    def __init__(self, instrument:str, base_path:str="data/processed", mode:str="decay_only", extension:str="wav" ):
        self.instrument = instrument
        self.base_path = base_path
        self.mode = mode
        self.extension = extension

    def generate_file_path(self, string_name:str, fret_no:str, exciter_type:str):
        return ('{base_path}/{mode}/{instrument}_{string_name}_{fret_no}_{exciter_type}.{extension}'
                .format(base_path=self.base_path,mode=self.mode,instrument=self.instrument,string_name=string_name,
                        fret_no=fret_no, exciter_type=exciter_type, extension=self.extension))

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
            case _:
                return TrainingParameters(batch_size=20000, epochs=200, learning_rate=1e-4, loss_function=torch.nn.MSELoss())

class TrainingDataGenerator:
    def generate_training_dataset(self, network_type:str, instrument:str):
        filepath_generator = FilepathGenerator(instrument=instrument)
        filepath = filepath_generator.generate_file_path('E', '0', 'plectrum')

        # WAV-Datei laden
        signal, samplerate = sf.read(filepath, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]

        # let's get the same delay we would use in the resonator loop
        resonator_factory = NeuralNetworkResonatorFactory()
        delay = resonator_factory.create_neural_network_delay(network_type)
        controls = resonator_factory.create_neural_network_controls(network_type)
        delay.set_base_frequency(83.05)

        # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
        # of delays is completely full and outputs the first sample
        max_delay_samples = 550
        init_samples = signal[:max_delay_samples]
        delay.prepare()
        delay.process_mono_split(init_samples) # output can be ignored
        remaining_signal = signal[max_delay_samples:]

        input_list = []
        target_list = []
        # iterate per sample over rest of audio file:
        # Feed the delay with the signal, the delay's output is the input of the neural network + control signal
        max_frames = 200000
        count = 0
        controls_input = controls.get_control_input_data()
        controls_row = controls_input.reshape(1, -1)
        for frame in frame_batch_generator(remaining_signal, 1):
            delay_output = delay.process_mono_split(frame)
            delay_row = delay_output.T # Transpose because trainingdata expects rows instead of columns

            inputs_concatenated = np.concatenate([delay_row, controls_row], axis=1)
            target_list.append(frame)
            input_list.append(inputs_concatenated)

            # force symmetric behaviour by adding symmetric training sample
            negative_delay_row = delay_row * -1
            negative_frame = frame * -1
            negative_inputs_concatenated = np.concatenate([negative_delay_row, controls_row], axis=1)
            target_list.append(negative_frame)
            input_list.append(negative_inputs_concatenated)

            count += 1
            if count > max_frames:
                break

        targets = torch.tensor(np.vstack(target_list), dtype=torch.float32)  # Shape: (N, 1)
        inputs = torch.tensor(np.vstack(input_list), dtype=torch.float32)
        return NeuralNetworkDataset(inputs, targets)
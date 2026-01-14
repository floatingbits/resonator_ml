import numpy as np
import soundfile as sf
import torch
import glob

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from resonator_ml.audio.util import frame_batch_generator
from resonator_ml.machine_learning.training import TrainingParameters
from resonator_ml.utility.random import ReproRNG

TRAINING_DATA_BASE_PATH = "data/processed"
TRAINING_DATA_SUB_PATH_DECAY = "decay_only"


def prepare_dataloader(dataset, batch_size=512):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

@dataclass
class TrainingFileDescriptor:
    model_name: str
    parameter_string: str
    file_name: str = "*.wav"

class TrainingFileFinder:
    def get_filepaths(self, file_descriptor: TrainingFileDescriptor) -> list[str]:
        file_pattern =  ('{base_path}/{mode}/{model_name}/{parameter_string}/{file_name}').format(
            base_path=TRAINING_DATA_BASE_PATH, mode=TRAINING_DATA_SUB_PATH_DECAY,
            model_name=file_descriptor.model_name, parameter_string=file_descriptor.parameter_string,
            file_name=file_descriptor.file_name)
        return glob.glob(file_pattern)



class FilepathGenerator:
    def __init__(self, instrument:str, base_path:str="data/processed", mode:str="decay_only", extension:str="wav" ):
        self.instrument = instrument
        self.base_path = base_path
        self.mode = mode
        self.extension = extension

    def generate_file_path(self, fret_no:str, exciter_type:str):
        return ('{base_path}/{mode}/{instrument}_{fret_no}_{exciter_type}.{extension}'
                .format(base_path=self.base_path,mode=self.mode,instrument=self.instrument,
                        fret_no=fret_no, exciter_type=exciter_type, extension=self.extension))


class TrainingDataGenerator:
    def __init__(self, training_parameters: TrainingParameters, training_file_descriptor: TrainingFileDescriptor, delay, controls):
        self.training_parameters = training_parameters
        self.training_file_descriptor = training_file_descriptor
        self.delay = delay
        self.controls = controls

    def generate_training_dataset(self):

        file_finder = TrainingFileFinder()
        file_paths = file_finder.get_filepaths(self.training_file_descriptor)

        accumulated_training_data = None
        for file_path in file_paths:
            training_data = self.generate_training_dataset_from_filepath(filepath=file_path)
            if not accumulated_training_data:
                accumulated_training_data = training_data
            else:
                accumulated_training_data.add(training_data)

        return accumulated_training_data

    def generate_training_dataloader(self):
        dataset = self.generate_training_dataset()
        return prepare_dataloader(dataset, batch_size=self.training_parameters.batch_size)


    def generate_training_dataset_from_filepath(self,  filepath: str):
        # WAV-Datei laden
        signal, samplerate = sf.read(filepath, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]



        # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
        # of delays is completely full and outputs the first sample
        max_delay_samples = 550
        init_samples = signal[:max_delay_samples]
        self.delay.prepare()
        self.delay.process_mono_split(init_samples)  # output can be ignored
        remaining_signal = signal[max_delay_samples:]

        input_list = []
        target_list = []
        # iterate per sample over rest of audio file:
        # Feed the delay with the signal, the delay's output is the input of the neural network + control signal
        max_frames = 80000
        count = 0
        controls_input = self.controls.get_control_input_data()
        controls_row = controls_input.reshape(1, -1)

        # reproducible random number generator to get reproducible results while randomly reducing training data
        rng = ReproRNG(100)
        for frame in frame_batch_generator(remaining_signal, 1):

            delay_output = self.delay.process_mono_split(frame)
            if rng.bernoulli(0.93):
                # Reduce Training data so that we can also use the tail of the file without slowing down training too much
                continue

            delay_row = delay_output.T  # Transpose because trainingdata expects rows instead of columns

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


class NeuralNetworkDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        windows: Tensor [N, n_inputs]
        targets: Tensor [N, 1]
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def add(self, dataset: 'NeuralNetworkDataset'):
        self.inputs = torch.vstack([self.inputs, dataset.inputs])
        self.targets = torch.vstack([self.targets, dataset.targets])

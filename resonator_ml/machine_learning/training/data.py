import math
from math import floor
from typing import Protocol

from sympy.physics.units import length
from torch.utils.data import Dataset

from resonator_ml.audio.core import MonoFrame, MonoSplitProcessor, MonoProcessor
import numpy as np
import soundfile as sf
import torch

from dataclasses import dataclass

from resonator_ml.utility.random import ReproRNG


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
        return self.inputs[idx], self.targets[idx], idx

    def add(self, dataset: 'NeuralNetworkDataset'):
        self.inputs = torch.vstack([self.inputs, dataset.inputs])
        self.targets = torch.vstack([self.targets, dataset.targets])


class ImplicitSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, wrapped_dataset: NeuralNetworkDataset, seq_len: int, hop: int):
        """
        X: [N, D]   (deine bisherigen Inputs)
        y: [N]
        """
        self.wrapped_dataset = wrapped_dataset
        self.K = seq_len
        self.hop = hop

    def __len__(self):
        return floor((len(self.wrapped_dataset) - self.K + 1) / self.hop)

    def __getitem__(self, idx):
        x_seq = self.wrapped_dataset.inputs[idx * self.hop : idx * self.hop + self.K]   # [K, D]
        y_seq = self.wrapped_dataset.targets[idx * self.hop : idx * self.hop + self.K]   # [K, 1]
        return x_seq, y_seq, idx

@dataclass
class AudioFeatureRequest:
    mono_signal: MonoFrame
    init_samples: int
    base_frequency: float
    sample_rate: float

@dataclass
class FeatureResponse:
    features: np.ndarray # shape (num_samples, num_features)

class AudioFeatureExtractor(Protocol):
    def extract(self, request: AudioFeatureRequest) -> FeatureResponse: ...

def init_samples(request: AudioFeatureRequest):
    return request.mono_signal[:request.init_samples]

def input_samples(request: AudioFeatureRequest):
    return request.mono_signal[request.init_samples:]

# e.g. for multi tap delay
class AudioMonoSplitFeatureExtractor(AudioFeatureExtractor):
    def __init__(self, audio_processor: MonoSplitProcessor):
        self.audio_processor = audio_processor
    def extract(self, request: AudioFeatureRequest) -> FeatureResponse:
        # init the processor (e.g. delays)
        self.audio_processor.process_mono_split(init_samples(request))
        output = self.audio_processor.process_mono_split(input_samples(request))
        return FeatureResponse(features=output.T)

# For simple targets (Audio File = Target)
class SimpleAudioFeatureExtractor(AudioFeatureExtractor):
    def extract(self, request: AudioFeatureRequest) -> FeatureResponse:
        output = input_samples(request)
        return FeatureResponse(features=output.T.reshape(-1,1))

class AudioDecayMeterFeatureExtractor(AudioFeatureExtractor):
    def __init__(self, decay_meter: MonoProcessor):
        self.decay_meter = decay_meter

    def extract(self, request: AudioFeatureRequest) -> FeatureResponse:
        self.decay_meter.prepare()
        self.decay_meter.process_mono(init_samples(request))

        decay_output = self.decay_meter.process_mono(input_samples(request))
        average_decay = decay_output.mean()
        # Map roughly +-0.04 scale to -1...1, with average at 0.
        decay_scale = 0.04
        decay_inertia = pow(0.5, request.base_frequency / request.sample_rate)
        # init
        smoothed_decay_value = float((decay_output[0] - average_decay) / decay_scale)
        decay_values = []
        for value in decay_output:
            new_decay_value = float((value - average_decay) / decay_scale)
            smoothed_decay_value = decay_inertia * smoothed_decay_value + (1 - decay_inertia) * new_decay_value
            feature_value = min(max(smoothed_decay_value, -1), 1)
            decay_values.append([feature_value]) # dataset with 1 Feature

        # apply minmax clipping only on actual training parameter, not before applying inertia.
        decay_rows = np.array(decay_values)
        # decay_target = np.array([[new_decay_parameter]])
        return FeatureResponse(features=decay_rows)


class ConstantControlValueFeatureProvider(AudioFeatureExtractor):
    def __init__(self, control_features: list[float]):
        self.control_features = control_features

    def extract(self, request: AudioFeatureRequest) -> FeatureResponse:
        row = np.asarray(self.control_features)[None, :]  # Shape (1, N)
        num_feature_rows = len(request.mono_signal) - request.init_samples
        control_features_expanded = np.repeat(row, num_feature_rows, axis=0)
        # might be replaced by arr = np.broadcast_to(row, (M, len(lst))) -> more memory efficient
        # (but read-only, which should be fine.)
        return FeatureResponse(features=control_features_expanded)

def combine_feature_responses(feature_responses: list[FeatureResponse]):
    feature_lists_to_concatenate = []
    for response in feature_responses:
        feature_lists_to_concatenate.append(response.features)

    n_samples = feature_lists_to_concatenate[0].shape[0] # num rows
    assert all(
        arr.ndim == 2 and arr.shape[0] == n_samples
        for arr in feature_lists_to_concatenate
    )

    return np.concatenate(feature_lists_to_concatenate, axis=1)

class TrainingDatasetReducer(Protocol):
    def reduce_dataset(self, dataset: NeuralNetworkDataset, num_output: int) -> NeuralNetworkDataset: ...

class TrainingDatasetManipulator(Protocol):
    def manipulate_dataset(self, dataset: NeuralNetworkDataset) -> NeuralNetworkDataset: ...

class TrainingDatasetSequencer:
    def sequence_dataset(self, dataset: NeuralNetworkDataset) -> NeuralNetworkDataset:
        sequenced_dataset = ImplicitSequenceDataset(dataset, 20, 20)
        inputs = []
        outputs = []
        for input, output, idx in sequenced_dataset:
            inputs.append(input)
            outputs.append(output)
        return NeuralNetworkDataset(inputs=torch.vstack(inputs), targets=torch.vstack(outputs))

class SymmetricVersionDatasetManipulator(TrainingDatasetManipulator):
    def __init__(self, input_indices: list[list[int]], output_indices: list[list[int]]):
        self.input_indices = input_indices
        self.output_indices = output_indices


    def manipulate_dataset(self, dataset: NeuralNetworkDataset) -> NeuralNetworkDataset:
        new_inputs = []
        new_outputs = []
        for inputs, outputs, idx in dataset:

            for index_config in self.input_indices:
                for i in range(index_config[0], index_config[1]):
                    inputs[i] = -inputs[i]
            for index_config in self.output_indices:
                for i in range(index_config[0], index_config[1]):
                    outputs[i] = -outputs[i]

            new_inputs.append(inputs)
            new_outputs.append(outputs)

        new_dataset = NeuralNetworkDataset(torch.vstack(new_inputs), torch.vstack(new_outputs))

        return new_dataset

class RandomAudioBurstDatasetManipulator(TrainingDatasetManipulator):
    def  __init__(self, indices: list[list[int]], manipulation_rate: float = 0.15, max_burst_size_ratio: float = 0.15, min_burst_size_ratio: float = 0.05, append: bool = True):
        self.manipulation_rate = manipulation_rate
        self.indices = indices # [[0,21]]
        self.max_burst_size_ratio = max_burst_size_ratio
        self.min_burst_size_ratio = min_burst_size_ratio
        self.append = append

    def manipulate_dataset(self, dataset: NeuralNetworkDataset) -> NeuralNetworkDataset:
        dataset_length = len(dataset)
        num_to_manipulate = int(dataset_length * self.manipulation_rate)
        new_inputs = []
        new_outputs = []
        for _ in range(num_to_manipulate):
            inputs, outputs, idx = dataset[np.random.randint(0,dataset_length)]
            inputs_positive = inputs.clone().detach()
            inputs_negative = inputs.clone().detach()
            for index_config in self.indices:
                burst = self.get_burst(inputs, index_config)
                # add positive and negative version of burst to prevent
                for i,index in enumerate(range(index_config[0], index_config[1])):
                    inputs_positive[index] += burst[i]
                    inputs_negative[index] -= burst[i]
            new_inputs.append(inputs_positive)
            new_outputs.append(outputs)
            new_inputs.append(inputs_negative)
            new_outputs.append(outputs)

        new_dataset = NeuralNetworkDataset(torch.vstack(new_inputs), torch.vstack(new_outputs))
        dataset.add(new_dataset)
        return dataset

    def get_burst(self, inputs, index_config):
        snippet_length = index_config[1] - index_config[0]

        ratio = (np.random.random_sample() * (self.max_burst_size_ratio - self.min_burst_size_ratio) +
                 self.min_burst_size_ratio)
        snippet_max = float("-inf")
        snippet_min = float("inf")
        for index in range(index_config[0], index_config[1]):
            value = inputs[index]
            if value > snippet_max:
                snippet_max = value
            if value < snippet_min:
                snippet_min = value
        burst_size = ratio * (snippet_max - snippet_min) / 2
        # Random burst lentgth
        increment = (np.random.random_sample() * 2 + 1) * 2 * math.pi / snippet_length
        phase = np.random.random_sample() * 2 * math.pi
        burst = []
        for index in range(index_config[0], index_config[1]):
            burst.append(burst_size * math.sin(phase))
            phase += increment

        return burst







class RandomTrainingDatasetReducer(TrainingDatasetReducer):
    def __init__(self, rng: ReproRNG):
        self.rng = rng
    def reduce_dataset(self, dataset: NeuralNetworkDataset, num_output: int) -> NeuralNetworkDataset:
        new_inputs = []
        new_outputs = []
        cur_len = len(dataset)
        p = float(num_output) / cur_len
        if p >= 1:
            return dataset # TODO issue warning

        for inputs, outputs, idx in dataset:
            if self.rng.bernoulli(p):
                new_outputs.append(outputs)
                new_inputs.append(inputs)

        return NeuralNetworkDataset(torch.vstack(new_inputs), torch.vstack(new_outputs))





class AudioTrainingDataGenerator:
    def __init__(self, base_frequency: float, input_feature_extractors: list[AudioFeatureExtractor], output_feature_extractors: list[AudioFeatureExtractor]):
        self.base_frequency = base_frequency
        self.input_feature_extractors = input_feature_extractors
        self.output_feature_extractors = output_feature_extractors
        pass

    def training_data_from_audio_file(self, file_path):
        signal, samplerate = sf.read(file_path, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]

        num_init_samples = 1100
        request = AudioFeatureRequest(mono_signal=signal, init_samples=num_init_samples,
                                      base_frequency=self.base_frequency, sample_rate=samplerate)

        input_feature_responses = []
        output_feature_responses = []
        for extractor in self.input_feature_extractors:
            input_feature_responses.append(extractor.extract(request))
        for extractor in self.output_feature_extractors:
            output_feature_responses.append(extractor.extract(request))

        combined_input = combine_feature_responses(input_feature_responses).astype(np.float32)
        combined_output = combine_feature_responses(output_feature_responses).astype(np.float32)

        targets = torch.from_numpy(combined_output) # Shape: (N, 1)
        inputs = torch.from_numpy(combined_input)
        return NeuralNetworkDataset(inputs, targets)




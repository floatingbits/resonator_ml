import abc
from abc import abstractmethod
from typing import Callable
import soundfile as sf
import torch
import torch.nn as nn

from resonator_ml.audio.core import MonoProcessor, MonoSplitProcessor, Tunable, MonoFrame, MultiChannelFrame, \
    MonoPushProcessor, MultiChannelPullProcessor
from resonator_ml.audio.delay_lines import SampleAccurateDelayLineMono

from dataclasses import dataclass

import numpy as np

from resonator_ml.machine_learning.training import TrainingParameters


class NeuralNetworkModule(nn.Module):
    def __init__(self, window_size=7, control_dim=0, hidden=64, n_hidden_layers:int = 1, activation:nn.Module=nn.Tanh()):
        super().__init__()

        self.in_dim = window_size + control_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
        )
        for n in range(n_hidden_layers):
            self.net.append(activation)
            if n < n_hidden_layers - 1:
                self.net.append(nn.Linear(hidden, hidden))

        self.net.append(nn.Linear(hidden, 1))
        # TODO this is only used as a storage for logging parameters. Define properly in config!
        self.hidden = hidden

    def forward(self, inputs):
        return self.net(inputs)


@dataclass
class DelayPattern:
    t_factor: float
    n_before: int
    n_after: int


class PatternDelayFactory:
    def __init__(self, delay_patterns: list[DelayPattern]):
        self.delay_patterns = delay_patterns

    def create_delays(self, sample_rate: int, base_time: float):
        result_delays = []
        for delay_pattern in self.delay_patterns:
            base_delay_time = delay_pattern.t_factor * base_time
            base_delay_samples = int(sample_rate * base_delay_time)
            sample_range = range(base_delay_samples - delay_pattern.n_before, base_delay_samples + delay_pattern.n_after + 1)
            for n_samples in sample_range:
                if n_samples > 0 :
                    result_delays.append(SampleAccurateDelayLineMono(n_samples, sample_rate))
        return  result_delays





class NnResonatorDelay(MonoSplitProcessor, MonoPushProcessor, MultiChannelPullProcessor, Tunable):
    def __init__(self, sample_rate: int, delay_factory: PatternDelayFactory):
        super().__init__(sample_rate)
        self.base_time = 1000
        self.delay_factory = delay_factory
        self.delays:list[SampleAccurateDelayLineMono] = []
        self.create_delays()

    def set_base_frequency(self, frequency: float):
        self.base_time = 1/frequency
        self.create_delays()

    def get_base_frequency(self) -> float:
        return 1/self.base_time

    def create_delays(self):
        self.delays = self.delay_factory.create_delays(self.sample_rate,self.base_time)

    def process_mono_split(self, samples: MonoFrame) -> MultiChannelFrame:
        return np.vstack([
            delay.process_mono(samples)
            for delay in self.delays
        ])

    def push_mono(self, mono):
        for delay in self.delays:
            delay.push_mono(mono)

    def pull_multi_channel(self, buffer_size: int) -> MultiChannelFrame:
        return np.vstack([
            delay.pull_mono(buffer_size)
            for delay in self.delays
        ])

class ControlInputProvider(abc.ABC):
    @abstractmethod
    def get_control_input_data(self) -> np.ndarray:
        pass

class DummyControlInputProvider(ControlInputProvider):
    def get_control_input_data(self) -> np.ndarray:
        return np.array([0])



class NeuralNetworkResonator(MonoProcessor):
    def __init__(self, model:NeuralNetworkModule, delay: NnResonatorDelay, controls: ControlInputProvider, sample_rate: int):
        super().__init__(sample_rate)
        self.model = model
        self.delay = delay
        self.controls = controls

    def process_mono(self, samples: MonoFrame) -> MonoFrame:
        # currently input samples are not used...
        out = np.zeros_like(samples)
        control_inputs = self.controls.get_control_input_data()
        for i, sample in enumerate(samples):
            delay_out = self.delay.pull_multi_channel(1) # sample by sample as long as we do not have a smarter implementation
            inputs = np.concatenate([delay_out.T[0], control_inputs], axis=0)

            out[i] = self.model.forward(torch.tensor(inputs, dtype=torch.float32))[0] # use the first output
            self.delay.push_mono(np.array([out[i]])) # push a single sample
        return out

class NNResonatorInitializer:
    def initialize(self, resonator: NeuralNetworkResonator, filepath):

        # WAV-Datei laden
        signal, samplerate = sf.read(filepath, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]

        # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
        # of delays is completely full and outputs the first sample
        max_delay_samples = 550
        init_samples = signal[:max_delay_samples]
        resonator.delay.prepare()
        resonator.delay.process_mono_split(init_samples)  # output can be ignored

@dataclass
class NeuralNetworkParameters:
    num_hidden_per_layer: int
    num_hidden_layers: int
    delay_patterns: list[DelayPattern]
    activation: nn.Module = nn.Tanh()


class NeuralNetworkResonatorFactory:
    def create_neural_network_resonator(self, sample_rate:int, parameters: NeuralNetworkParameters):
        delay = self.create_neural_network_delay(sample_rate, parameters)
        controls = self.create_neural_network_controls(parameters)
        return NeuralNetworkResonator(
            self.create_neural_network_module(delay, controls, parameters=parameters),
            delay,
            controls,
            sample_rate
        )

    def create_neural_network_module(self, delay, controls, parameters: NeuralNetworkParameters):
        return NeuralNetworkModule(len(delay.delays), len(controls.get_control_input_data()), n_hidden_layers=parameters.num_hidden_layers, activation=parameters.activation, hidden=parameters.num_hidden_per_layer)

    def create_neural_network_delay(self, sample_rate: int, parameters: NeuralNetworkParameters):
        return NnResonatorDelay(sample_rate, PatternDelayFactory(parameters.delay_patterns))

    def create_neural_network_controls(self, parameters: NeuralNetworkParameters):
        return DummyControlInputProvider()


class Trainer:
    def __init__(self, training_parameters: TrainingParameters):
        self.training_parameters = training_parameters

    def train_neural_network(self, model, dataloader, device="cpu",
                             epoch_callback: Callable[[int, int, float], None]=None):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_parameters.learning_rate)

        for epoch in range(self.training_parameters.epochs):
            epoch_loss = 0.0

            for x_input, y_target in dataloader:
                x_input = x_input.to(device)
                y_target = y_target.to(device)

                # Vorwärts
                y_pred = model(x_input)

                # Loss im Sample-Bereich
                loss = self.training_parameters.loss_function(y_pred, y_target)

                # Rückwärts
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(x_input)

            if epoch_callback:
                epoch_callback(epoch, self.training_parameters.epochs, epoch_loss / len(dataloader.dataset))

        return model



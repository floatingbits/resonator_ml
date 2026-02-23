import abc
from abc import abstractmethod
from typing import Callable
import soundfile as sf
import torch
import torch.nn as nn

from resonator_ml.audio.core import MonoProcessor, MonoSplitProcessor, Tunable, MonoFrame, MultiChannelFrame, \
    MonoPushProcessor, MultiChannelPullProcessor
from resonator_ml.audio.delay_lines import SampleAccurateDelayLineMono, SampleAccurateMultiHeadDelay

from dataclasses import dataclass

import numpy as np

from resonator_ml.machine_learning.training.analysis import PerSampleLossTracker
from resonator_ml.machine_learning.training.parameters import TrainingParameters


class NeuralNetworkModule(nn.Module):
    def __init__(self, window_size=7, control_dim=0, hidden=64, n_hidden_layers:int = 1, activation:nn.Module=nn.Tanh()):
        super().__init__()

        self.in_dim = window_size + control_dim
        self.common_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
        )
        for n in range(n_hidden_layers):
            self.common_net.append(activation)
            if n < n_hidden_layers - 1:
                self.common_net.append(nn.Linear(hidden, hidden))


        self.audio_head = nn.Linear(hidden, 1)
        self.decay_head = nn.Linear(hidden, 1)
        # TODO this is only used as a storage for logging parameters. Define properly in config!
        self.hidden = hidden

    def forward(self, inputs):
        return self.forward_audio_only(inputs)
    def forward_audio_only(self, inputs):
        return self.audio_head(self.common_net(inputs))
    def forward_with_decay(self, inputs):
        common_out = self.common_net(inputs)
        return self.audio_head(common_out), self.decay_head(common_out)


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
    def create_delay_times(self, sample_rate: int, base_time: float):
        result_delay_times = []
        for delay_pattern in self.delay_patterns:
            base_delay_time = delay_pattern.t_factor * base_time
            base_delay_samples = int(sample_rate * base_delay_time)
            sample_range = range(base_delay_samples - delay_pattern.n_before, base_delay_samples + delay_pattern.n_after + 1)
            for n_samples in sample_range:
                if n_samples > 0 :
                    result_delay_times.append(n_samples)
        return  result_delay_times





class NnResonatorDelay(MonoSplitProcessor, MonoPushProcessor, MultiChannelPullProcessor, Tunable):
    def __init__(self, sample_rate: int, delay_factory: PatternDelayFactory):
        super().__init__(sample_rate)
        self.base_time = 1000
        self.delay_factory = delay_factory
        self.delay_times:list[int] = []
        self.delay = None
        self.create_delays()

    def set_base_frequency(self, frequency: float):
        self.base_time = 1/frequency
        self.create_delays()

    def get_base_frequency(self) -> float:
        return 1/self.base_time

    def create_delays(self):
        self.delay_times = self.delay_factory.create_delay_times(self.sample_rate, self.base_time)
        self.delay = SampleAccurateMultiHeadDelay(self.delay_times, self.sample_rate)

    def process_mono_split(self, samples: MonoFrame) -> MultiChannelFrame:
        return self.delay.process_mono_split(samples)

    def push_mono(self, mono):
        self.delay.push_mono(mono)

    def pull_multi_channel(self, buffer_size: int) -> MultiChannelFrame:
        return self.delay.pull_multi_channel(buffer_size)

class ControlInputProvider(abc.ABC):
    @abstractmethod
    def get_control_input_data(self) -> np.ndarray:
        pass

class DummyControlInputProvider(ControlInputProvider):
    def get_control_input_data(self) -> np.ndarray:
        return np.array([0])



class NeuralNetworkResonator(MonoProcessor):
    def __init__(self, model:NeuralNetworkModule, delay: NnResonatorDelay, controls: ControlInputProvider, sample_rate: int, use_decay_feature: bool):
        super().__init__(sample_rate)
        self.model = model
        self.delay = delay
        self.controls = controls
        self.use_decay_feature = use_decay_feature

    # def process_mono(self, samples: MonoFrame) -> MonoFrame:
    #     # currently input samples are not used but for the length of the output...
    #     out = np.zeros_like(samples)
    #     control_inputs = self.controls.get_control_input_data()
    #     for i, sample in enumerate(samples):
    #         delay_out = self.delay.pull_multi_channel(1) # sample by sample as long as we do not have a smarter implementation
    #         concatenation_array = [delay_out.T[0], control_inputs]
    #         if self.use_decay_feature:
    #             concatenation_array.append(np.array([0]))
    #         inputs = np.concatenate(concatenation_array, axis=0)
    #
    #         out[i] = self.model.forward(torch.tensor(inputs, dtype=torch.float32)).detach()[0] # use the first output
    #         self.delay.push_mono(np.array([out[i]])) # push a single sample
    #     return out

    def process_mono(self, samples: MonoFrame) -> MonoFrame:
        total_len = len(samples)
        control_inputs = self.controls.get_control_input_data()

        out_chunks = []
        processed = 0

        while processed < total_len:
            block_size = min(
                self.delay.delay.n_samples_in_buffer,
                min(self.delay.delay_times),
                total_len - processed,
            )

            # shape: (n_channels, block_size)
            delay_out = self.delay.pull_multi_channel(block_size)

            # → (block_size, n_channels)
            delay_features = delay_out.T

            # Control-Inputs batchen
            control_block = np.repeat(
                control_inputs[np.newaxis, :],
                block_size,
                axis=0
            )

            feature_blocks = [delay_features, control_block]

            if self.use_decay_feature:
                decay_feature = np.zeros((block_size, 1))
                feature_blocks.append(decay_feature)

            # (block_size, total_feature_dim)
            inputs = np.concatenate(feature_blocks, axis=1)

            torch_in = torch.tensor(inputs, dtype=torch.float32)
            torch_out = self.model.forward(torch_in).detach().numpy()

            # mono output → erster Outputkanal
            out_block = torch_out[:, 0]

            out_chunks.append(out_block)

            # Block push
            self.delay.push_mono(out_block)

            processed += block_size

        return np.concatenate(out_chunks, axis=0)

class NNResonatorInitializer:
    def initialize(self, resonator: NeuralNetworkResonator, filepath):

        # WAV-Datei laden
        signal, samplerate = sf.read(filepath, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]

        # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
        # of delays is completely full and outputs the first sample
        max_delay_samples = 2*550
        init_samples = signal[:max_delay_samples]
        resonator.delay.prepare()
        resonator.delay.process_mono_split(init_samples)  # output can be ignored

@dataclass
class NeuralNetworkParameters:
    num_hidden_per_layer: int
    num_hidden_layers: int
    delay_patterns: list[DelayPattern]
    activation: nn.Module = nn.Tanh()
    use_decay_feature: bool = False


class NeuralNetworkResonatorFactory:
    def create_neural_network_resonator(self, sample_rate:int, parameters: NeuralNetworkParameters):
        delay = self.create_neural_network_delay(sample_rate, parameters)
        controls = self.create_neural_network_controls(parameters)
        return NeuralNetworkResonator(
            self.create_neural_network_module(delay, controls, parameters=parameters),
            delay,
            controls,
            sample_rate,
            parameters.use_decay_feature
        )

    def create_neural_network_module(self, delay, controls, parameters: NeuralNetworkParameters):
        num_inputs = len(delay.delay_times)
        if parameters.use_decay_feature:
            num_inputs += 1
        return NeuralNetworkModule(num_inputs, len(controls.get_control_input_data()), n_hidden_layers=parameters.num_hidden_layers, activation=parameters.activation, hidden=parameters.num_hidden_per_layer)

    def create_neural_network_delay(self, sample_rate: int, parameters: NeuralNetworkParameters):
        return NnResonatorDelay(sample_rate, PatternDelayFactory(parameters.delay_patterns))

    def create_neural_network_controls(self, parameters: NeuralNetworkParameters):
        return DummyControlInputProvider()

def forward_sequence(model, x_seq):
    """
    x_seq: [B, K, D]
    """
    B, K, D = x_seq.shape
    x_flat = x_seq.reshape(B * K, D)
    y_hat_flat = model(x_flat)        # [B*K]
    y_hat_seq = y_hat_flat.reshape(B, K,1)
    return y_hat_seq


class Trainer:
    def __init__(self, training_parameters: TrainingParameters, model_path: str):
        self.training_parameters = training_parameters
        self.model_path = model_path

    def train_neural_network(self, model, dataloader, device="cpu",
                             epoch_callback: Callable[[int, int, float, float, float], None]=None):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_parameters.learning_rate)

        best_training = float("inf")

        torch.set_printoptions(sci_mode=True)
        dataset_len = len(dataloader.dataset)
        tracker = PerSampleLossTracker()
        decay_weight = 0.05 # lambda
        for epoch in range(self.training_parameters.epochs):
            epoch_loss = 0.0
            max_batch_loss = 0.0
            min_batch_loss = float("inf")
            for x_input, y_target, ids in dataloader:
                x_input = x_input.to(device)
                y_target = y_target.to(device)
                if x_input.ndim == 3:
                    y_pred = forward_sequence(model, x_input)
                else:
                    y_pred = model(x_input)

                # audio_pred, decay_pred = model(x_input)
                # loss_audio = self.training_parameters.loss_function(audio_pred, audio_target, x_input).mean(dim=1)
                # loss_decay = self.training_parameters.decay_loss_function(decay_pred, decay_target)
                # per_sample_loss = loss_audio + decay_weight * loss_decay
                # loss = per_sample_loss.mean()

                per_sample_loss = self.training_parameters.loss_function(y_pred, y_target, x_input).mean(dim=1)
                loss = per_sample_loss.mean()


                # Rückwärts
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                epoch_loss += batch_loss * len(x_input)
                if batch_loss > max_batch_loss:
                    max_batch_loss = batch_loss
                if batch_loss < min_batch_loss:
                    min_batch_loss = batch_loss

                # tracker.update(ids, per_sample_loss, y_pred=y_pred)

            # worst_samples = tracker.worst_samples(k=10, by="quantile", q=0.95)
            # for idx, loss, prediction, max_prediction, min_prediction, last_prediction in worst_samples:
            #     print(idx, "Loss: ", loss, "Prediction(mean, max, min):", prediction, max_prediction, min_prediction, last_prediction)
            #     print(dataloader.dataset.__getitem__(idx))

            loss_average = epoch_loss / dataset_len
            # TODO: Use validation loss
            if loss_average < best_training:
                best_training = loss_average
                torch.save(model.state_dict(), self.model_path)
            if epoch_callback:
                epoch_callback(epoch, self.training_parameters.epochs, loss_average, min_batch_loss, max_batch_loss)

        print("Autocorrelation of sample loss between epochs")
        for sid, _, _ in tracker.persistent_hard_samples(k=10):
            print(sid, tracker.loss_autocorrelation(sid))


        return model



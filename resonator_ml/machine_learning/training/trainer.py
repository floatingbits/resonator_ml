import dataclasses
from typing import Callable
from abc import ABC, abstractmethod
import torch

from typing import Generic, TypeVar

from torch.optim import Optimizer, AdamW

from resonator_ml.machine_learning.custom_loss_functions import log_spectral_loss, energy_loss, relative_l1, \
    correlation_loss, dynamic_sign_loss, dynamic_sign_loss_soft
from resonator_ml.machine_learning.loop_filter.neural_network import forward_sequence, flatten
from resonator_ml.machine_learning.training.analysis import PerSampleLossTracker
from resonator_ml.machine_learning.training.parameters import TrainingParameters

class LossModule(ABC):
    @abstractmethod
    def compute_loss(self, y_pred, batch):
        pass

class WeightedComposedLossModule(LossModule):
    def __init__(self, weighted_sub_modules: list[tuple[float, LossModule]]):
        self.weighted_sub_modules = weighted_sub_modules

    def compute_loss(self, y_pred, batch):
        loss = None
        for weight, loss_module in self.weighted_sub_modules:
            current_loss = weight * loss_module.compute_loss(y_pred, batch)
            # TODO is 0 + current_loss allowed?
            if loss:
                loss += current_loss
            else:
                loss = current_loss
        return loss

class TimeSpectralEnergyLossModule(LossModule):
    def __init__(self, spectral_lambda, energy_lambda, phase_lambda, correlation_lambda, correlation_loss_threshold):
        self.spectral_lambda = spectral_lambda
        self.energy_lambda = energy_lambda
        self.phase_lambda = phase_lambda
        self.correlation_lambda = correlation_lambda
        self.correlation_loss_threshold = correlation_loss_threshold

    def compute_loss(self, y_pred, batch):
        x_input, y_target, ids = batch
        #phase_loss = relative_l1(flatten(y_pred), flatten(y_target), flatten(x_input)).mean(dim=0).mean()
        phase_loss = relative_l1(y_pred, y_target, x_input).mean(dim=0).mean()
        spec_loss = log_spectral_loss(y_pred, y_target)
        e_loss = energy_loss(y_pred, y_target).mean()
        c_loss = dynamic_sign_loss_soft(y_pred,y_target, self.correlation_loss_threshold) # prevent negative phase solution
        loss = self.spectral_lambda * spec_loss + self.energy_lambda * e_loss + self.phase_lambda * phase_loss + self.correlation_lambda * c_loss
        return loss


class Predictor(ABC):
    @abstractmethod
    def predict(self, batch):
        pass

class StaticPredictor(Predictor):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def predict(self, batch):
        x_input, y_target, ids = batch
        x_input = x_input.to(self.device)
        y_target = y_target.to(self.device)
        if x_input.ndim == 3:
            y_pred = forward_sequence(self.model, x_input)
        else:
            y_pred = self.model(x_input)
        return y_pred


class OptimizationPolicy(ABC):
    def backprop(self, loss):
        pass

class SimpleOptimizationPolicy(OptimizationPolicy):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TrainingStep(ABC):
    @abstractmethod
    def step(self, batch):
        pass

class LossTracker:
    def __init__(self):
        self.accumulated_loss = 0
        self.accumulated_weight = 0
        self.min_batch_loss = float('inf')
        self.max_batch_loss = float('-inf')
        self.best_epoch = float('inf')
        self.after_best_epoch_hooks = []

    def reset_epoch(self):
        self.accumulated_loss = 0
        self.accumulated_weight = 0
        self.min_batch_loss = float('inf')
        self.max_batch_loss = float('-inf')

    def reset_all(self):
        self.best_epoch = float("inf")
        self.reset_epoch()

    def track_batch_loss(self, loss, weight):
        val = loss.item()
        self.accumulated_loss += val * weight
        self.accumulated_weight += weight
        if self.max_batch_loss < val:
            self.max_batch_loss = val
        if self.min_batch_loss > val:
            self.min_batch_loss = val
    def get_average(self):
        return self.accumulated_loss / self.accumulated_weight

    def end_epoch(self):
        average = self.get_average()
        self.reset_epoch()
        if average < self.best_epoch:
            self.best_epoch = average
            for h in self.after_best_epoch_hooks:
                h()

class ModelPersistenceManager(ABC):
    @abstractmethod
    def save(self):
        pass
class SimpleModelPersistenceManager(ModelPersistenceManager):
    def __init__(self, save_path, model):
        self.model = model
        self.save_path = save_path

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)

class Validator:
    def __init__(self, loss_module: LossModule, test_data):
        self.loss_module = loss_module
        self.test_data = test_data
    def calculate_validation_loss(self, predictor: Predictor):
        with torch.no_grad():
            y_pred = predictor.predict(self.test_data)
        return self.loss_module.compute_loss(y_pred, self.test_data)


class ComposedTrainingStep(TrainingStep):
    def __init__(self, predictor: Predictor,
                 loss_module: LossModule,
                 optimization_policy: OptimizationPolicy):
        self.predictor = predictor
        self.loss_module = loss_module
        self.optimization_policy = optimization_policy
        self.before_forward_hooks = []
        self.after_forward_hooks = []
        self.after_loss_hooks = []
        self.after_backward_hooks = []

    def step(self, batch):
        for h in self.before_forward_hooks:
            h(batch)

        y_pred = self.predictor.predict(batch)

        for h in self.after_forward_hooks:
            h(y_pred)

        loss = self.loss_module.compute_loss(y_pred,batch)

        for h in self.after_loss_hooks:
            h(loss, batch)

        self.optimization_policy.backprop(loss)

        for h in self.after_backward_hooks:
            h()


class Trainer:
    def __init__(self, training_parameters: TrainingParameters, model_path: str, step_strategy):
        self.training_parameters = training_parameters
        self.model_path = model_path
        self.step_strategy = step_strategy

    def train_neural_network(self, model, dataloader, device="cpu",
                             epoch_callback: Callable[[int, int, float, float, float], None]=None):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_parameters.learning_rate)
  #      optimizer = torch.optim.SGD(model.parameters(), lr=self.training_parameters.learning_rate)
        best_training = float("inf")

        torch.set_printoptions(sci_mode=True)
        dataset_len = len(dataloader.dataset)
        tracker = PerSampleLossTracker()
        decay_weight = 0.05 # lambda
        energy_lambda = self.training_parameters.energy_lambda
        spectral_lambda = self.training_parameters.spectral_lambda
        phase_lambda = self.training_parameters.phase_lambda
        lambda_sum = spectral_lambda + energy_lambda + phase_lambda
        energy_lambda /= lambda_sum
        spectral_lambda /= lambda_sum
        phase_lambda /= lambda_sum

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

                # per_sample_loss = self.training_parameters.loss_function(y_pred, y_target, x_input).mean(dim=1)
                # loss = per_sample_loss.mean()
                phase_loss = self.training_parameters.loss_function(flatten(y_pred), flatten(y_target),
                                                                    flatten(x_input)).mean(dim=0).mean()
                spec_loss = log_spectral_loss(y_pred, y_target)
                e_loss = energy_loss(y_pred, y_target).mean()
                loss = spectral_lambda * spec_loss + energy_lambda * e_loss + phase_lambda * phase_loss

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

class OuterTrainer:
    def __init__(self, step_strategy: TrainingStep, dataloader):
        self.step_strategy = step_strategy
        self.dataloader = dataloader
        self.after_epoch_hooks = []

    def train_neural_network(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch()
            for h in self.after_epoch_hooks:
                h(epoch)

    def train_epoch(self, ):
        for batch in self.dataloader:
            self.step_strategy.step(batch)

@dataclasses.dataclass
class EpochEvent:
    epoch: int
    training_loss: float
    validation_loss: float


class AfterEpochListener(ABC):
    @abstractmethod
    def after_epoch(self, event: EpochEvent):
        pass

T = TypeVar("T")
class ComposedAfterEpochListener(AfterEpochListener, Generic[T]):
    def __init__(self, condition: Callable[[EpochEvent], bool], converter: Callable[[EpochEvent], T], action: Callable[[T], None]):
        self.condition = condition
        self.converter = converter
        self.action = action

    def after_epoch(self, event: EpochEvent):
        if self.condition(event):
            argument = self.converter(event)
            self.action(argument)


def generate_epoch_to_float_converter_ramp(min_epoch, max_epoch, min_value:float, max_value:float)-> Callable[[EpochEvent], float]:
    def ramp(event: EpochEvent):
        if event.epoch <= min_epoch:
            return min_value
        if event.epoch >= max_epoch:
            return max_value
        return min_value + (max_value - min_value)*(event.epoch - min_epoch)/(max_epoch - min_epoch)

    return ramp


def generate_min_epoch_condition(min_epoch) -> Callable[[EpochEvent], bool]:
    def condition(event: EpochEvent):
        return event.epoch >  min_epoch
    return condition

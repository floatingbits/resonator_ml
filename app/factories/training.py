import dataclasses

import torch
from torch.utils.data import DataLoader, Dataset

from app.adapters.series_provider import TrainingLossSeriesProvider
from app.config.app import Config
from app.factories.storage import file_storage

from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkModule
from resonator_ml.machine_learning.training.trainer import OptimizationPolicy, ComposedTrainingStep, OuterTrainer, \
    Predictor, StaticPredictor, TimeSpectralEnergyLossModule, LossModule, \
    LossTracker, ModelPersistenceManager, Validator, AfterEpochListener, ComposedAfterEpochListener, generate_min_epoch_condition, \
    generate_epoch_to_float_converter_ramp, EpochEvent


@dataclasses.dataclass
class TrainingRunContext:
    model: NeuralNetworkModule
    optimizer: torch.optim.Optimizer
    optimization_policy: OptimizationPolicy
    predictor: Predictor
    loss_module: LossModule
    persistence_manager: ModelPersistenceManager
    total_num_epochs: int
    training_dataloader: DataLoader
    validation_dataset: Dataset
    loss_module_schedulers: list[AfterEpochListener]

def build_predictor(model, device="cpu"):
    return StaticPredictor(model=model, device=device)


def print_callback(epoch: int, epochs: int, epoch_loss: float, min_batch_loss: float, max_batch_loss: float, validation_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f} Validation Loss {validation_loss:.10f} Min_Batch_Loss: {min_batch_loss:.10f} Max_Batch_Loss: {max_batch_loss:.10f}")

def trainer(context: TrainingRunContext):
    # return Trainer(training_parameters(config), model_path=file_storage(config).model_file_path().as_posix())
    step_strategy = ComposedTrainingStep(predictor=context.predictor,
                                         loss_module=context.loss_module,
                                         optimization_policy=context.optimization_policy)
    loss_tracker = LossTracker()
    def best_epoch_hook():
        context.persistence_manager.save()
    loss_tracker.after_best_epoch_hooks.append(best_epoch_hook)

    def track_loss(loss, batch):
        x,y,idx = batch
        loss_tracker.track_batch_loss(loss, len(x))


    step_strategy.after_loss_hooks.append(track_loss)
    validator = Validator(context.loss_module, context.validation_dataset)

    def after_epoch_hook(epoch):
        average = loss_tracker.get_average()
        validation_loss = validator.calculate_validation_loss(context.predictor)
        print_callback(epoch,
                       context.total_num_epochs, epoch_loss=average,
                       min_batch_loss=loss_tracker.min_batch_loss, max_batch_loss=loss_tracker.max_batch_loss, validation_loss=validation_loss.item())
        loss_tracker.end_epoch()
        epoch_event = EpochEvent(epoch,average,validation_loss.item)
        for listener in context.loss_module_schedulers:
            listener.after_epoch(epoch_event)
        # if context.total_num_epochs == epoch + 1:
        #     context.persistence_manager.save()

    outer_trainer =  OuterTrainer(step_strategy, context.training_dataloader)
    outer_trainer.after_epoch_hooks.append(after_epoch_hook)
    return outer_trainer


def loss_module(config: Config):
    return TimeSpectralEnergyLossModule(
        spectral_lambda=config.training_parameters.spectral_lambda,
        energy_lambda=config.training_parameters.energy_lambda,
        phase_lambda=config.training_parameters.phase_lambda,
        correlation_lambda=config.training_parameters.correlation_lambda,
        correlation_loss_threshold=config.training_parameters.correlation_loss_threshold)

def dynamic_loss_scheduler(config: Config, loss_mod: TimeSpectralEnergyLossModule):
    def correlation_action(val: float):
        print('Setting correlation weight to', val)
        loss_mod.correlation_lambda = val
    def spectral_action(val: float):
        print('Setting spectral weight to', val)
        loss_mod.spectral_lambda = val
    def energy_action(val: float):
        print('Setting energy weight to', val)
        loss_mod.energy_lambda = val

    return [
        ComposedAfterEpochListener(
            generate_min_epoch_condition(15),
            generate_epoch_to_float_converter_ramp(15, 20, config.training_parameters.correlation_lambda,config.training_parameters.correlation_lambda/80),
            correlation_action
        ),
        # ComposedAfterEpochListener(
        #     generate_min_epoch_condition(20),
        #     generate_epoch_to_float_converter_ramp(15, 30, config.training_parameters.spectral_lambda,
        #                                            config.training_parameters.spectral_lambda * 2),
        #     spectral_action
        # ),
        # ComposedAfterEpochListener(
        #     generate_min_epoch_condition(20),
        #     generate_epoch_to_float_converter_ramp(15, 30, config.training_parameters.energy_lambda,
        #                                            config.training_parameters.energy_lambda / 2),
        #     energy_action
        # ),
    ]


def training_parameters(config: Config):
    return config.training_parameters


def training_loss_series_provider(config: Config):
    return TrainingLossSeriesProvider(file_storage(config).output_folder_base_path())

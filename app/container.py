from app.adapters.file_storage import LocalFileSystemStorage, DictJsonFileLogger
from app.adapters.series_provider import TrainingLossSeriesProvider
from app.config.app import Config
from resonator_ml.machine_learning.file_management import create_loop_filter_model_file_name
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonatorFactory, Trainer, \
    NNResonatorInitializer
from resonator_ml.machine_learning.loop_filter.training_data import TrainingFileDescriptor, TrainingDataGenerator, \
    FilepathGenerator, TrainingFileFinder, TrainingDatasetCache
from resonator_ml.machine_learning.training import TrainingParameterFactory
import torch

from resonator_ml.ports.file_storage import FileStorage, DictStorage


def app_config():
    return Config()

def training_loss_series_provider():
    return TrainingLossSeriesProvider(file_storage().model_file_path().parent.parent)

def training_parameters():
    config = app_config()
    training_parameter_factory = TrainingParameterFactory()
    return training_parameter_factory.create_parameters(config.training_parameters_version)


def nn_resonator(load_model_weights: bool, initialize_resonator: bool):
    config = app_config()
    resonator_factory = NeuralNetworkResonatorFactory()
    resonator =  resonator_factory.create_neural_network_resonator(config.resonator_version, config.sample_rate)
    # let's get the same delay we would use in the resonator loop
    resonator.delay.set_base_frequency(config.base_frequency)

    # TODO: business logic, get out of wiring
    if load_model_weights:
        model = resonator.model
        model.load_state_dict(torch.load(file_storage().model_file_path(), weights_only=True))
        model.eval()

    if initialize_resonator:
        filepaths = training_file_paths()
        filepath = filepaths[0]
        initializer = NNResonatorInitializer()
        initializer.initialize(resonator, filepath)


    return resonator



def trainer():
    return Trainer(training_parameters())

def training_file_descriptor():
    config = app_config()

    return TrainingFileDescriptor(model_name=config.instrument_name, parameter_string='0')

def training_data_cache():
    config = app_config()
    return TrainingDatasetCache(file_storage().training_data_cache_path().absolute().as_posix())

def training_data_generator():
    resonator = nn_resonator(load_model_weights=False, initialize_resonator=False)
    return TrainingDataGenerator(training_parameters(),
                                                    training_file_descriptor=training_file_descriptor(),
                                                    delay=resonator.delay, controls=resonator.controls,
                                 training_dataset_cache=training_data_cache())

def training_file_paths():
    file_descriptor = training_file_descriptor()
    file_finder = TrainingFileFinder()
    return file_finder.get_filepaths(file_descriptor)


def out_filepath():
    config = app_config()
    filepath_generator = FilepathGenerator(instrument=config.instrument_name)
    # TODO: is this used? Refactor
    filepath_generator.base_path = 'data/results'
    filepath_generator.mode = 'decay_only/workspace'
    return filepath_generator.generate_file_path('0', 'plectrum')


def file_storage() -> FileStorage:
    return LocalFileSystemStorage(app_config())

def parameters_storage() -> DictStorage:

    return DictJsonFileLogger(file_storage().parameters_output_path())
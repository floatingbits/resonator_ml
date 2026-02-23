from app.adapters.file_storage import LocalFileSystemStorage, DictJsonFileLogger
from app.adapters.series_provider import TrainingLossSeriesProvider
from app.config.app import Config
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonatorFactory, Trainer, \
    NNResonatorInitializer
from resonator_ml.machine_learning.loop_filter.training_data import TrainingFileDescriptor, TrainingDataGenerator, \
    FilepathGenerator, TrainingFileFinder, TrainingDatasetCache, TrainingDataCacheKeyProvider, \
    TrainingDataCacheKeyGenerator

import torch

from resonator_ml.ports.file_storage import FileStorage, DictStorage


def app_config():
    return Config()

def training_loss_series_provider(config: Config):
    return TrainingLossSeriesProvider(file_storage(config).output_folder_base_path())

def training_parameters(config: Config):
    return config.training_parameters


def nn_resonator(config: Config, load_model_weights: bool, initialize_resonator: bool):

    resonator_factory = NeuralNetworkResonatorFactory()
    resonator =  resonator_factory.create_neural_network_resonator(config.sample_rate, config.neural_network_parameters)
    # let's get the same delay we would use in the resonator loop
    resonator.delay.set_base_frequency(config.base_frequency)

    # TODO: business logic, get out of wiring
    if load_model_weights:
        model = resonator.model
        model.load_state_dict(torch.load(file_storage(config).model_file_path(), weights_only=True))
        model.eval()

    if initialize_resonator:
        filepath = init_sound_file(config)
        initializer = NNResonatorInitializer()
        initializer.initialize(resonator, filepath)


    return resonator

def init_sound_file(config: Config):
    if config.initialize_sound_file_path:
        filepath = config.initialize_sound_file_path
    else:
        filepaths = training_file_paths(config)
        filepath = filepaths[0]
    return filepath



def trainer(config: Config):
    return Trainer(training_parameters(config), model_path=file_storage(config).model_file_path().as_posix())

def training_file_descriptor(config: Config):

    return TrainingFileDescriptor(model_name=config.instrument_name, parameter_string='0')

def training_data_cache(config: Config):
    return TrainingDatasetCache(file_storage(config).training_data_cache_path().absolute().as_posix(),
                                cache_key_provider=training_data_cache_key_provider(config))

def training_data_generator(config: Config):
    resonator = nn_resonator(config, load_model_weights=False, initialize_resonator=False)
    return TrainingDataGenerator(training_parameters(config),
                                                    training_file_descriptor=training_file_descriptor(config),
                                                    delay=resonator.delay, controls=resonator.controls,
                                 training_dataset_cache=training_data_cache(config),
                                 base_frequency=config.base_frequency)
def training_data_cache_key_generator(config: Config):
    return TrainingDataCacheKeyGenerator()

def training_data_cache_key_provider(config: Config):
    return TrainingDataCacheKeyProvider(config,training_data_cache_key_generator(config))

def training_file_paths(config: Config):
    file_descriptor = training_file_descriptor(config)
    file_finder = TrainingFileFinder()
    return file_finder.get_filepaths(file_descriptor)


def out_filepath(config: Config):
    filepath_generator = FilepathGenerator(instrument=config.instrument_name)
    # TODO: is this used? Refactor
    filepath_generator.base_path = 'data/results'
    filepath_generator.mode = 'decay_only/workspace'
    return filepath_generator.generate_file_path('0', 'plectrum')


def file_storage(config: Config) -> FileStorage:
    return LocalFileSystemStorage(config)

def parameters_storage(config: Config) -> DictStorage:

    return DictJsonFileLogger(file_storage(config).parameters_output_path())
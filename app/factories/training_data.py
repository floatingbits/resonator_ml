from app.config.app import Config
from app.factories.training import training_parameters
from app.factories.resonator import nn_resonator
from app.factories.storage import file_storage
from resonator_ml.machine_learning.loop_filter.training_data import TrainingFileFinder, TrainingDataCacheKeyProvider, \
    TrainingDataCacheKeyGenerator, TrainingDataGenerator, TrainingDatasetCache, TrainingFileDescriptor


def training_file_paths(config: Config):
    file_descriptor = training_file_descriptor(config)
    file_finder = TrainingFileFinder()
    return file_finder.get_filepaths(file_descriptor)


def training_data_cache_key_provider(config: Config):
    return TrainingDataCacheKeyProvider(config,training_data_cache_key_generator(config))


def training_data_cache_key_generator(config: Config):
    return TrainingDataCacheKeyGenerator()


def init_sound_file(config: Config):
    if config.initialize_sound_file_path:
        filepath = config.initialize_sound_file_path
    else:
        filepaths = training_file_paths(config)
        filepath = filepaths[0]
    return filepath


def training_data_generator(config: Config):
    resonator = nn_resonator(config)
    return TrainingDataGenerator(training_parameters(config),
                                                    training_file_descriptor=training_file_descriptor(config),
                                                    delay=resonator.delay, controls=resonator.controls,
                                 training_dataset_cache=training_data_cache(config),
                                 base_frequency=config.base_frequency)


def training_data_cache(config: Config):
    return TrainingDatasetCache(file_storage(config).training_data_cache_path().absolute().as_posix(),
                                cache_key_provider=training_data_cache_key_provider(config))


def training_file_descriptor(config: Config):

    return TrainingFileDescriptor(model_name=config.instrument_name, parameter_string='0')

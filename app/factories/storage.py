from app.adapters.file_storage import LocalFileSystemStorage, DictJsonFileLogger
from app.config.app import Config
from resonator_ml.machine_learning.loop_filter.training_data import FilepathGenerator
from resonator_ml.ports.file_storage import FileStorage, DictStorage


def file_storage(config: Config) -> FileStorage:
    return LocalFileSystemStorage(config)


def parameters_storage(config: Config) -> DictStorage:
    return DictJsonFileLogger(file_storage(config).parameters_output_path())

def results_storage(config: Config) -> DictStorage:
    return DictJsonFileLogger(file_storage(config).results_output_path())


def out_filepath(config: Config):
    filepath_generator = FilepathGenerator(instrument=config.instrument_name)
    # TODO: is this used? Refactor
    filepath_generator.base_path = 'data/results'
    filepath_generator.mode = 'decay_only/workspace'
    return filepath_generator.generate_file_path('0', 'plectrum')

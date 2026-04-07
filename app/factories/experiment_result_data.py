from app.adapters.evaluation_data_provider import ExperimentDataProvider
from app.config.app import Config
from app.factories.storage import file_storage


def experiment_result_data_provider(config: Config):
    return ExperimentDataProvider(file_storage(config).results_output_path().parent.parent)

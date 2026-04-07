from app.config.app import Config as AppConfig
from app.factories.experiment import experiment_runner, experiment_definition


def run(config: AppConfig):
    experiment_runner(config).run(experiment_definition(config))


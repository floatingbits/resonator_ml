import random

from app.config.app import Config as AppConfig
from app.factories.auto_train import auto_trainer
from app.factories.experiment import experiment_runner, experiment_definition


def run(config: AppConfig):
    def config_extractor(conf):
        return {
                "seq_len": conf.training_parameters.seq_length,
                "spectral": conf.training_parameters.spectral_lambda,
                "energy": conf.training_parameters.energy_lambda,
                "phase": conf.training_parameters.phase_lambda,
                "lr": conf.training_parameters.learning_rate,
            }
    def determine_break_condition(c, result_metrics):
        return c >= 20
    direction = True
    def modify_config(conf: AppConfig, result_metrics: dict|None = None, has_improved = True):
        nonlocal direction
        if not has_improved:
            direction = not direction
        conf.training_parameters.epochs = 10
        #conf.training_parameters.seq_length += random.choice([10,20,-10,-20])
        conf.training_parameters.spectral_lambda /= 1.5
        conf.training_parameters.energy_lambda /= 1.5
        #conf.training_parameters.phase_lambda *= random.randint(8,12)/10
        conf.training_parameters.correlation_lambda = 0
        conf.training_parameters.learning_rate /= 2
        return conf
    auto_trainer(config).run(config_extractor, determine_break_condition, modify_config)


from app.config.app import Config
from app.factories.storage import file_storage
from app.orchestration.experiment.runner import ExperimentRunner
from app.orchestration.experiment.definition import ExperimentDefinition
import copy
import numpy as np

def experiment_definition(config:Config):
    configs = []
    match config.experiment_name:
        case "test_experiment":
            for lr in np.linspace(1.0e-6, 1.5e-5, 5):
                config1 = copy.deepcopy(config)
                config1.training_parameters.learning_rate = lr
                configs.append(config1)

            return ExperimentDefinition(3,configs, lambda conf: {"learn_rate": conf.training_parameters.learning_rate})
        case "loss_balance":
            # What is the best combination of loss values?
            # What is making the training most stable? What is making the resulting recursive system most stable?
            # This experiment delivers some optima around energy:6, spectral: 1.5, phase: 5, but this might change according to other parameters
            # for energy in [1,4,8]:
            #     for spectral in [1, 2, 6]:
            #         for phase in [1, 3, 5]:
            for energy in [3,6,8]:
                for spectral in [0.8, 1.5, 5]:
                    for phase in [2, 5]:
                        config1 = copy.deepcopy(config)
                        config1.training_parameters.energy_lambda = energy
                        config1.training_parameters.phase_lambda = phase
                        config1.training_parameters.spectral_lambda = spectral
                        configs.append(config1)

            return ExperimentDefinition(15, configs, lambda conf: {
                "spectral": conf.training_parameters.spectral_lambda,
                "energy": conf.training_parameters.energy_lambda,
                "phase": conf.training_parameters.phase_lambda
            })
        case "neurons_amount_1":
            # Do higher amounts of neurons deliver better results?
            # => No benefit yet, we see lower lrs might be needed for higher neuron amounts, because loss values
            # start to jump around more
            for n_neurons in [30, 60, 120,200,500,1000,5000]:
                config1 = copy.deepcopy(config)
                config1.neural_network_parameters.num_hidden_per_layer = n_neurons
                configs.append(config1)

            return ExperimentDefinition(15, configs, lambda conf: {
                "n_hidden": conf.neural_network_parameters.num_hidden_per_layer,
            })
        case "neurons_amount_vs_lr":
            # DO higher amounts of hidden neurons need lower learning rates?
            # Yes, but still, there is no real benefit in higher neuron amounts
            # More experiments to come, when some other questions are solved (Risk of getting stuck in non-sense plateaus)
            for n_neurons in [ 600, 2000]:
                for lr in [6.e-7,9.e-7,3e-6]:
                    config1 = copy.deepcopy(config)
                    config1.neural_network_parameters.num_hidden_per_layer = n_neurons
                    config1.training_parameters.learning_rate = lr
                    config1.training_parameters.epochs = 65
                    configs.append(config1)

            return ExperimentDefinition(5, configs, lambda conf: {
                "n_hidden": conf.neural_network_parameters.num_hidden_per_layer,
                "lr": conf.training_parameters.learning_rate,
            })
        case "synthetic_training":
            # Do we still need synthetic training samples to stabilize our results?
            # => Clearly no
            config1 = copy.deepcopy(config)

            config2 = copy.deepcopy(config)
            config2.training_parameters.training_data_glob = "[0-9].wav"
            configs.append(config2)
            configs.append(config1)

            return ExperimentDefinition(15, configs, lambda conf: {
                "glob": conf.training_parameters.training_data_glob
            })
        case "correlation_loss":

            for c_loss in [1.6,1.8,2,2.3,2.6]:
                config1 = copy.deepcopy(config)
                config1.training_parameters.correlation_lambda = c_loss
                config1.training_parameters.epochs = 10
                configs.append(config1)

            return ExperimentDefinition(8, configs, lambda conf: {
                "c_lambda": conf.training_parameters.correlation_lambda
            })
        case "correlation_loss_threshold":
            # what is the best combination for reducing the risk of running into a negative phase loss high plateau?
            # => high c_loss and closer to zero thresholds are safest, but  need to be combined with loss scheduling
            # in later epochs to not corrupt the actual regularization
            # for c_loss in [1.5, 2, 2.6]:
            #    for c_threshold in [-0.1,-0.2,-0.3]:
            # for c_loss in [2.8, 3.4, 4]:
            #     for c_threshold in [-0.35, -0.45, -0.55]:
            for c_loss in [10, 14, 20]:
                for c_threshold in [-0.2, -0.4, -0.5]:
                    config1 = copy.deepcopy(config)
                    config1.training_parameters.correlation_lambda = c_loss
                    config1.training_parameters.correlation_loss_threshold = c_threshold
                    config1.training_parameters.epochs = 10
                    configs.append(config1)

            return ExperimentDefinition(10, configs, lambda conf: {
                "c_lambda": conf.training_parameters.correlation_lambda,
                "c_threshold": conf.training_parameters.correlation_loss_threshold
            })
        case "seq_len":
            # is there an ideal seq_len under my current conditions?
            # => It seems there is an optimum around 150
            # for seq_len in [40, 80, 150]:
            for seq_len in [40,80,150,200, 300]:
                config1 = copy.deepcopy(config)
                config1.training_parameters.seq_length = seq_len
                config1.training_parameters.epochs = 45
                configs.append(config1)
            return ExperimentDefinition(5, configs, lambda conf: {
                "seq_len": conf.training_parameters.seq_length
            })
        case "seq_len_vs_loss_lambdas":
            # does seq_len influence ideal loss balance?
            for seq_len in [40, 180]:
                for energy in [3,6]:
                    for spectral in [1, 3]:
                        for phase in [2, 5]:
                            config1 = copy.deepcopy(config)
                            config1.training_parameters.energy_lambda = energy
                            config1.training_parameters.phase_lambda = phase
                            config1.training_parameters.spectral_lambda = spectral
                            config1.training_parameters.seq_length = seq_len
                            config1.training_parameters.epochs = 40
                            configs.append(config1)

            return ExperimentDefinition(4, configs, lambda conf: {
                "seq_len": conf.training_parameters.seq_length,
                "spectral": conf.training_parameters.spectral_lambda,
                "energy": conf.training_parameters.energy_lambda,
                "phase": conf.training_parameters.phase_lambda
            })
        case "loss_balance_2":


            for spectral in [0.1, 2]:
                for phase in [0.5, 1.5]:
                    config1 = copy.deepcopy(config)
                    config1.training_parameters.phase_lambda = phase
                    config1.training_parameters.spectral_lambda = spectral
                    config1.training_parameters.epochs = 40
                    configs.append(config1)

            return ExperimentDefinition(12, configs, lambda conf: {
                "spectral": conf.training_parameters.spectral_lambda,
                "energy": conf.training_parameters.energy_lambda,
                "phase": conf.training_parameters.phase_lambda
            })
        case "seq_len_vs_loss_lambdas_2":
            for seq_len in [180, 200, 220 ]:
                for spectral in [0.5, 0.8]:
                    for phase in [1.5, 1.8]:
                        for energy in [2.5, 2.8]:
                            config1 = copy.deepcopy(config)
                            config1.training_parameters.phase_lambda = phase
                            config1.training_parameters.spectral_lambda = spectral
                            config1.training_parameters.energy_lambda = energy
                            config1.training_parameters.seq_length = seq_len
                            config1.training_parameters.epochs = 35
                            configs.append(config1)

            return ExperimentDefinition(8, configs, lambda conf: {
                "seq_len": conf.training_parameters.seq_length,
                "spectral": conf.training_parameters.spectral_lambda,
                "energy": conf.training_parameters.energy_lambda,
                "phase": conf.training_parameters.phase_lambda
            })

        case _:
            raise ValueError("Experiment name {experiment} not found.".format(experiment=config.experiment_name))

def experiment_runner(config:Config):
    return ExperimentRunner(file_storage=file_storage(config))


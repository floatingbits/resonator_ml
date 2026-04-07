import torch

from app.config.app import Config


from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonatorFactory


def nn_resonator(config: Config):

    resonator_factory = NeuralNetworkResonatorFactory()
    resonator =  resonator_factory.create_neural_network_resonator(config.sample_rate, config.neural_network_parameters)
    # let's get the same delay we would use in the resonator loop
    resonator.delay.set_base_frequency(config.base_frequency)


    return resonator

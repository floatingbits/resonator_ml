import pytest
from resonator_ml.machine_learning.loop_filter.neural_network import NnResonatorDelay, PatternDelayFactory, \
    DelayPattern, DummyControlInputProvider, NeuralNetworkModule
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonator
import numpy as np
def test_resonator_throws_no_exception():

    sample_rate = 40000
    fundamental_frequency = 2000
    delay_patterns = [DelayPattern(1,1,1)]
    delay_factory = PatternDelayFactory(delay_patterns)
    delay = NnResonatorDelay(sample_rate,delay_factory)
    delay.set_base_frequency(fundamental_frequency)
    controls = DummyControlInputProvider()
    delay_times = delay_factory.create_delay_times(sample_rate=sample_rate, base_time=fundamental_frequency/sample_rate)
    model = NeuralNetworkModule(window_size=len(delay_times), control_dim=len(controls.get_control_input_data()))
    resonator = NeuralNetworkResonator(delay=delay, controls=DummyControlInputProvider(), sample_rate=sample_rate,
                                       use_decay_feature=False, model=model)
    samples = np.zeros(10, dtype=np.float32)
    resonator.process_mono(samples)
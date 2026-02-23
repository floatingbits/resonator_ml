from resonator_ml.audio.metering import StatefulEnergyMeter, DecayMeter
import numpy as np


class TestStatefulEnergyMeter:
    def test_empty_outputs_zero_until_full(self):
        window_size = 10
        meter = StatefulEnergyMeter(window_size=window_size, sample_rate=123)
        samples = np.ones(window_size, dtype=np.float32)
        output = meter.process_mono(samples)
        for value in output:
            assert value == 0
        # window is now full
        output = meter.process_mono(samples)
        # value is now window_size * 1²
        for value in output:
            assert value == window_size
        output = meter.process_mono(samples*0.5)
        # value is now window_size * 0.5²
        for value in output:
            assert window_size > value >= pow(0.5, 2) * window_size
        output[9] = pow(0.5, 2) * window_size


class TestDecayMeter:
    def test_simple_decay(self):
        window_size = 10
        meter = DecayMeter(window_size, 123, 10)
        samples = np.ones(window_size, dtype=np.float32)
        output = meter.process_mono(samples)
        # output should be all 0 (delayed energy is 0 because of empty window + delay is still in its initial 0 buffer)
        # , but is kind of an edge case tied to the implementation detail, so do not assert
        output = meter.process_mono(samples)
        # output should be all 0 (delayed energy is 0 because of empty window)
        # , but is kind of an edge case tied to the implementation detail, so do not assert
        output = meter.process_mono(samples)
        for value in output:
            assert value == 0 # we filled the meter only with 1s, so simply no decay
        samples = np.ones(window_size, dtype=np.float32) * 0.5
        output = meter.process_mono(samples)
        for value in output:
            assert 0 > value >= -0.60206
        # Last value should be pretty close to -0.60205999 or log10(0.25)
        assert output[9] < -0.602
import numpy as np
from resonator_ml.audio.delay_lines import SampleAccurateDelayLineMono, SampleAccurateMultiHeadDelay
import pytest

class TestSampleAccurateDelayLineMono:
    def test_process_mono(self):
        samples = np.zeros(10, dtype=np.float32)
        samples[0] = 1.0
        delay_line = SampleAccurateDelayLineMono(6, 123)
        delay_line.prepare()
        output = delay_line.process_mono(samples)
        assert output[6] == 1
        assert output[5] == 0
        assert output[7] == 0
        assert output[0] == 0


class TestSampleAccurateMultiHeadDelay:
    def test_process_mono_split(self):
        samples = np.zeros(10, dtype=np.float32)
        samples[0] = 1.0
        delay_line = SampleAccurateMultiHeadDelay([5,3], 123)
        delay_line.prepare()
        output = delay_line.process_mono_split(samples)
        assert output[0][5] == 1
        assert output[1][5] == 0
        assert output[0][0] == 0
        assert output[1][3] == 1

    def test_push_pull(self):
        samples = np.zeros(10, dtype=np.float32)
        samples[0] = 1.0
        delay_line = SampleAccurateMultiHeadDelay([5,3], 123)
        delay_line.prepare()

        delay_line.push_mono(samples)
        output = delay_line.pull_multi_channel(len(samples))

        assert output[0][5] == 1
        assert output[1][5] == 0
        assert output[0][0] == 0
        assert output[1][3] == 1

    def test_buffer_assertion_underrun(self):
        delay_times_in_samples = [5, 3]
        delay_line = SampleAccurateMultiHeadDelay(delay_times_in_samples, 123)
        delay_line.prepare()
        samples = np.zeros(10, dtype=np.float32)
        samples[0] = 1.0
        delay_line.push_mono(samples)
        max_allowed_pull_from_buffer = len(samples) + min(delay_times_in_samples)
        output = delay_line.pull_multi_channel(max_allowed_pull_from_buffer)
        with pytest.raises(AssertionError, match="Buffer underrun"):
            output = delay_line.pull_multi_channel(1)







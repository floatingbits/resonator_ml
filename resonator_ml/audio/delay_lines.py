import numpy as np
from .core import MonoProcessor, MonoPushProcessor, MonoPullProcessor, MonoFrame, MonoSplitProcessor, \
    MultiChannelPullProcessor, MultiChannelFrame, BufferedProcessor
from abc import ABC


class BaseDelay(ABC):
    def __init__(self, delay_ms: float):
        self.delay_ms = delay_ms

    def set_delay_ms(self, delay_ms: float):
        self.delay_ms = delay_ms


class FractionalDelayMono(MonoProcessor, BaseDelay):
    def __init__(self, max_delay_ms: float, delay_ms: float, sample_rate: int):
        super(MonoProcessor, self).__init__(sample_rate)
        super(BaseDelay, self).__init__(delay_ms)
        self.max_delay_ms = max_delay_ms
        # will be defined in prepare
        self.max_delay_samples = None
        self.write_index = None
        self.buffer = None

    def prepare(self):
        super().prepare()

        # max Delay in Samples
        self.max_delay_samples = int(self.sample_rate * self.max_delay_ms / 1000)

        # Four additional samples for Interpolation
        self.buffer = np.zeros(self.max_delay_samples + 4, dtype=np.float32)
        self.write_index = 2  # for

    def reset(self):
        self.buffer[:] = 0
        self.write_index = 2


    # -------------------------------
    # Lagrange 3rd order Interpolator
    # -------------------------------
    def _lagrange3(self, x0, x1, x2, x3, frac):
        """
        frac: fractional delay offset in [0, 1)
        x0 = y[n-1]
        x1 = y[n]
        x2 = y[n+1]
        x3 = y[n+2]
        """
        c0 = (-frac * (frac - 1) * (frac - 2)) / 6.0
        c1 = ((frac + 1) * (frac - 1) * (frac - 2)) / 2.0
        c2 = (- (frac + 1) * frac * (frac - 2)) / 2.0
        c3 = ((frac + 1) * frac * (frac - 1)) / 6.0

        return c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3

    def _read_fractional(self, delay_samples: float) -> float:
        buffer_len = len(self.buffer)

        read_pos = self.write_index - delay_samples

        # Wraparound
        while read_pos < 1:
            read_pos += buffer_len

        i1 = int(np.floor(read_pos)) % buffer_len  # y[n]
        frac = read_pos - np.floor(read_pos)

        # Indices for four-sample-Lagrange
        i0 = (i1 - 1) % buffer_len  # y[n-1]
        i2 = (i1 + 1) % buffer_len  # y[n+1]
        i3 = (i1 + 2) % buffer_len  # y[n+2]

        return self._lagrange3(
            self.buffer[i0], self.buffer[i1],
            self.buffer[i2], self.buffer[i3],
            frac
        )

    def process_mono(self, mono: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mono)
        buffer_len = len(self.buffer)

        # Delaytime in Samples
        delay_samples = self.delay_ms * self.sample_rate / 1000.0
        delay_samples = np.clip(delay_samples, 1.0, self.max_delay_samples - 2)

        for i, x in enumerate(mono):
            y = self._read_fractional(delay_samples)
            out[i] = y

            self.buffer[self.write_index] = x

            self.write_index = (self.write_index + 1) % buffer_len

        return out


class SampleAccurateDelayLineMono(MonoProcessor, MonoPushProcessor, MonoPullProcessor):
    def __init__(self, n_delay_samples: int, sample_rate: int):
        super().__init__(sample_rate)
        self.n_delay_samples = n_delay_samples
        self.buffer = np.zeros(self.n_delay_samples, dtype=np.float32)
        # enable pull/delayed push with separate write/read indices
        self.write_index = 0
        self.read_index = 0

    def prepare(self):
        super().prepare()
        self.write_index = 0
        self.read_index = 0

    def reset(self):
        self.buffer[:] = 0
        self.write_index = 0
        self.read_index = 0

    def process_mono(self, mono) -> MonoFrame:
        out = np.zeros_like(mono)
        for i, sample in enumerate(mono):
            out[i] = self.buffer[self.read_index]
            self.buffer[self.write_index] = sample
            self.write_index = self.write_index + 1
            self.read_index = self.read_index + 1
            if self.write_index == self.n_delay_samples:
                self.write_index = 0
            if self.read_index == self.n_delay_samples:
                self.read_index = 0
        return out

    def push_mono(self, mono):
        for i, sample in enumerate(mono):
            self.buffer[self.write_index] = sample
            self.write_index = self.write_index + 1
            if self.write_index == self.n_delay_samples:
                self.write_index = 0


    def pull_mono(self, buffer_size: int) -> MonoFrame:
        out = np.zeros(buffer_size, dtype=np.float32)
        for i in range(buffer_size):
            out[i] = self.buffer[self.read_index]
            self.read_index = self.read_index + 1
            if self.read_index == self.n_delay_samples:
                self.read_index = 0
        return out


class SampleAccurateMultiHeadDelay(MonoSplitProcessor, MonoPushProcessor, MultiChannelPullProcessor, BufferedProcessor):


    def __init__(self, delay_times_in_samples: list[int], sample_rate: int, delay_buffer_size: int = 8192):
        super().__init__(sample_rate)
        super(BufferedProcessor).__init__()
        super(MonoSplitProcessor).__init__()
        super(MonoPushProcessor).__init__()
        super(MultiChannelPullProcessor).__init__()
        self.delay_times_in_samples = delay_times_in_samples
        self.delay_buffer_size = delay_buffer_size
        self.buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)
        # enable pull/delayed push with separate write/read indices
        self.write_index = 0
        self.read_indices = []
        self.reset_indices()

    def prepare(self):
        super().prepare()
        self.reset_indices()

    def reset_indices(self):
        self.write_index = 0
        self.read_indices = [
            self.delay_buffer_size - cur_delay_time - 1
            for cur_delay_time in self.delay_times_in_samples
        ]
        # consider buffer filled with zeroes initially
        self.n_samples_in_buffer = min(self.delay_times_in_samples)

    def reset(self):
        self.buffer[:] = 0
        self.reset_indices()

    # def process_mono_split(self, mono) -> MonoFrame:
    #     out = np.zeros_like(mono)
    #     for i, sample in enumerate(mono):
    #         for channel,read_index in enumerate(self.read_indices):
    #             out[channel][i] = self.buffer[read_index]
    #             self.read_indices[channel] = read_index + 1
    #             if self.read_indices[channel] == self.max_delay_samples:
    #                 self.read_indices[channel] = 0
    #
    #         self.buffer[self.write_index] = sample
    #         self.write_index = self.write_index + 1
    #         if self.write_index == self.max_delay_samples:
    #             self.write_index = 0
    #
    #     return out

    def push_mono(self, mono):
        self.push_buffer(len(mono))
        for i, sample in enumerate(mono):
            self.buffer[self.write_index] = sample
            self.write_index = self.write_index + 1
            if self.write_index == self.delay_buffer_size:
                self.write_index = 0

    def pull_multi_channel(self, buffer_size: int) -> MultiChannelFrame:
        self.pull_buffer(buffer_size)
        out = np.zeros((len(self.read_indices), buffer_size), dtype=np.float32)
        for i in range(buffer_size):
            for channel, _ in enumerate(self.read_indices):
                self.read_indices[channel] = self.read_indices[channel] + 1
                if self.read_indices[channel] == self.delay_buffer_size:
                    self.read_indices[channel] = 0
                out[channel][i] = self.buffer[self.read_indices[channel]]


        return out

    def process_mono_split(self, samples: MonoFrame) -> MultiChannelFrame:
        buffer_size = samples.shape[0]
        cur_pos = 0
        out = np.zeros((len(self.read_indices),0),dtype=np.float32)
        while buffer_size:
            cur_buffer_size = min(buffer_size, self.delay_buffer_size)
            next_pos = cur_pos + cur_buffer_size
            cur_samples = samples[cur_pos:next_pos]
            self.push_mono(cur_samples)
            cur_output = self.pull_multi_channel(cur_buffer_size)
            out = np.hstack([out, cur_output])
            buffer_size -= cur_buffer_size
            cur_pos = next_pos
        return out
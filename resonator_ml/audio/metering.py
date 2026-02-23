from .core import MonoProcessor, MonoFrame

import numpy as np
import math
from .delay_lines import SampleAccurateDelayLineMono
from scipy.signal import stft

class StatefulEnergyMeter(MonoProcessor):
    def __init__(self, window_size: int, sample_rate: int):
        super().__init__(sample_rate)
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.buffer_pos = 0
        self.energy = 0.0
        self.filled = 0  # wie viele Samples schon im Buffer sind

    def prepare(self):
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.energy = 0.0
        self.filled = 0

    def process_mono(self, samples: MonoFrame) -> MonoFrame:
        samples = np.asarray(samples, dtype=np.float32)
        output = np.empty_like(samples)

        for i, x in enumerate(samples):
            x2 = x * x

            if self.filled < self.window_size:
                # Fenster noch nicht voll → nur aufbauen
                self.buffer[self.buffer_pos] = x
                self.energy += x2
                self.filled += 1

                output[i] = 0.0
            else:
                # Sliding Window
                old = self.buffer[self.buffer_pos]
                self.energy -= old * old

                self.buffer[self.buffer_pos] = x
                self.energy += x2

                output[i] = self.energy

            self.buffer_pos = (self.buffer_pos + 1) % self.window_size

        return output

class DecayMeter(MonoProcessor):
    def __init__(self, window_size: int, sample_rate: int, time_in_samples: int):
        super().__init__(sample_rate)
        self.delay = SampleAccurateDelayLineMono(time_in_samples, sample_rate)
        self.energy_meter = StatefulEnergyMeter(window_size,sample_rate)

    def prepare(self):
        self.delay.prepare()
        self.energy_meter.prepare()

    def process_mono(self, samples: MonoFrame) -> MonoFrame:
        energy = self.energy_meter.process_mono(samples)
        delayed_energy_samples = self.delay.process_mono(energy)
        output = np.empty_like(samples)
        for i, delayed_energy in enumerate(delayed_energy_samples):
            cur_energy = energy[i]
            if delayed_energy == 0:
                output[i] = 0
            else:
                if cur_energy == 0:
                    output[i] = float('-inf')
                else:
                    output[i] = -1*math.log(delayed_energy/cur_energy, 10)

        return output


def compute_spectrogram(x, fs, n_fft=1024, hop=256):
    f, t, zxx = stft(
        x,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window="hann",
        padded=False,
        boundary=None,
    )

    magnitude_db = 20 * np.log10(np.abs(zxx) + 1e-12)
    return f, t, magnitude_db


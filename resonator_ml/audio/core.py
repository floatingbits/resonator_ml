from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Callable

import numpy as np
MonoFrame = np.ndarray   # shape: (samples,)
MultiChannelFrame = np.ndarray  # shape: (channels, samples)

class BaseProcessor(ABC):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._prepared = False

    def prepare(self):
        """
        Setup internal state based on sample_rate.
        Called once before processing.
        """
        self._prepared = True

    def reset(self):
        """Reset internal DSP state (delay buffers, filter memory, envelopes...)."""

class MonoPushProcessor(ABC):
    @abstractmethod
    def push_mono(self, mono):
        pass

class MonoPullProcessor(ABC):

    @abstractmethod
    def pull_mono(self, buffer_size: int) -> MonoFrame:
        pass


class MultiChannelPullProcessor(ABC):

    @abstractmethod
    def pull_multi_channel(self, buffer_size: int) -> MultiChannelFrame:
        pass


class Tunable(ABC):

    @abstractmethod
    def set_base_frequency(self, frequency: float):
        pass

    @abstractmethod
    def get_base_frequency(self) -> float:
        pass


class AudioProcessor(BaseProcessor):

    @abstractmethod
    def process(self, buffer: MultiChannelFrame) -> MultiChannelFrame:
        pass

class MonoProcessor(BaseProcessor):
    @abstractmethod
    def process_mono(self, samples: MonoFrame) -> MonoFrame:
        """
        Process a mono signal. Shape: (samples)
        """
        pass

class MonoSplitProcessor(BaseProcessor):
    @abstractmethod
    def process_mono_split(self, samples: MonoFrame) -> MultiChannelFrame:
        """
        Process a mono signal. Shape: (samples)
        """
        pass


class MultiChannelProcessor(AudioProcessor):
    def __init__(self, mono_factory: Callable[[int], MonoProcessor], sample_rate: int):
        super().__init__(sample_rate)
        self.mono_factory = mono_factory
        self.processors: list[MonoProcessor] = []

    def prepare(self):
        super().prepare()
        # Re-create processors with correct sample_rate
        self.processors = [self.mono_factory(self.sample_rate)]

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        audio: (channels, samples)
        """
        channels = audio.shape[0]

        # adjust processor list if channel count changed
        if len(self.processors) != channels:
            self.processors = [
                self.mono_factory(self.sample_rate)
                for _ in range(channels)
            ]

        return np.vstack([
            proc.process_mono(audio[ch])
            for ch, proc in enumerate(self.processors)
        ])
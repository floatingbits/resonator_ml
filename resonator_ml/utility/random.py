import numpy as np

class ReproRNG:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def randint(self, n: int) -> int:
        """Integer in [0, n]"""
        return int(self.rng.integers(0, n + 1))

    def bernoulli(self, p: float) -> bool:
        """True with probability p"""
        return bool(self.rng.random() < p)

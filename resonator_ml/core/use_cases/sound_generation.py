import soundfile as sf
import time
import numpy as np


def print_callback(epoch: int, epochs: int, epoch_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f}")


class GenerateSoundFile:
    def __init__(self, resonator,  out_filepath: str, samplerate: int):
        self.resonator = resonator
        self.out_filepath = out_filepath
        self.samplerate = samplerate

    def execute(self):

        start = time.time()

        dummy_input = np.zeros(20 * self.samplerate, dtype=np.float32)
        out = self.resonator.process_mono(dummy_input)

        sf.write(self.out_filepath, out, self.samplerate)

        end = time.time()
        length = end - start
        # Show the results : this can be altered however you like
        print("It took", length, "seconds!")
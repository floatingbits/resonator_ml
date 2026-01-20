import soundfile as sf
import time
import numpy as np

from resonator_ml.ports.file_storage import FileStorage


def print_callback(epoch: int, epochs: int, epoch_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f}")


class GenerateSoundFile:
    def __init__(self, resonator,  file_storage: FileStorage, samplerate: int):
        self.resonator = resonator
        self.file_storage = file_storage
        self.samplerate = samplerate

    def execute(self):

        start = time.time()

        dummy_input = np.zeros(20 * self.samplerate, dtype=np.float32)
        out = self.resonator.process_mono(dummy_input)
        out_path = self.file_storage.sound_output_path()
        out_path.parent.mkdir(parents=True,exist_ok=True)
        sf.write(out_path, out, self.samplerate)

        end = time.time()
        length = end - start
        # Show the results : this can be altered however you like
        print("It took", length, "seconds!")
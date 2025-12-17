import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import time


from resonator_ml.machine_learning.file_management import create_loop_filter_model_file_name
from resonator_ml.machine_learning.loop_filter.neural_network import NeuralNetworkResonatorFactory
from resonator_ml.machine_learning.loop_filter.training_data import  FilepathGenerator

if __name__ == "__main__":
    instrument = "KS"
    resonator_type_name = "v1"
    model_suffix = ""
    model_suffix = "_2"
    model_name = instrument + "_" + resonator_type_name  + model_suffix


    resonator_factory = NeuralNetworkResonatorFactory()

    start = time.time()

    # Modell erstellen
    resonator = resonator_factory.create_neural_network_resonator(resonator_type_name, sample_rate=44100)
    model = resonator.model
    model.load_state_dict(torch.load(create_loop_filter_model_file_name(model_name), weights_only=True))
    model.eval()

    delay = resonator.delay
    delay.set_base_frequency(83.05)

    filepath_generator = FilepathGenerator(instrument=instrument)
    filepath = filepath_generator.generate_file_path('E', '0', 'plectrum')

    # WAV-Datei laden
    signal, samplerate = sf.read(filepath, dtype='float32')
    if signal.ndim == 2:
        signal = signal[:, 0]

    # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
    # of delays is completely full and outputs the first sample
    max_delay_samples = 550
    init_samples = signal[:max_delay_samples]
    delay.prepare()
    delay.process_mono_split(init_samples)  # output can be ignored

    dummy_input = np.zeros(20 * 44100, dtype=np.float32)
    out = resonator.process_mono(dummy_input)

    filepath_generator.base_path = 'data/results'
    filepath_generator.mode = 'decay_only/workspace'
    filepath = filepath_generator.generate_file_path('E', '0', 'plectrum')

    sf.write(filepath, out, samplerate)

    end = time.time()
    length = end - start
    # Show the results : this can be altered however you like
    print("It took", length, "seconds!")




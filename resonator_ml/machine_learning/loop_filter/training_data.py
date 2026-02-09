import numpy as np
import soundfile as sf
import torch
import glob
from pathlib import Path

from dataclasses import dataclass

from torch.utils.data import DataLoader

from resonator_ml.audio.metering import DecayMeter
from resonator_ml.audio.util import frame_batch_generator
from resonator_ml.machine_learning.training.data import NeuralNetworkDataset, AudioTrainingDataGenerator, \
    AudioMonoSplitFeatureExtractor, ConstantControlValueFeatureProvider, AudioDecayMeterFeatureExtractor, \
    SimpleAudioFeatureExtractor, RandomTrainingDatasetReducer, RandomAudioBurstDatasetManipulator, \
    SymmetricVersionDatasetManipulator, SequenceDataset
from resonator_ml.machine_learning.training.parameters import TrainingParameters
from resonator_ml.utility.random import ReproRNG

TRAINING_DATA_BASE_PATH = "data/processed"
TRAINING_DATA_SUB_PATH_DECAY = "decay_only"


def prepare_dataloader(dataset, batch_size=512):
    sequence_dataset = SequenceDataset(wrapped_dataset=dataset, seq_len=20)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

@dataclass
class TrainingFileDescriptor:
    model_name: str
    parameter_string: str
    file_name: str = "*.wav"

class TrainingFileFinder:
    def get_filepaths(self, file_descriptor: TrainingFileDescriptor) -> list[str]:
        file_pattern =  ('{base_path}/{mode}/{model_name}/{parameter_string}/{file_name}').format(
            base_path=TRAINING_DATA_BASE_PATH, mode=TRAINING_DATA_SUB_PATH_DECAY,
            model_name=file_descriptor.model_name, parameter_string=file_descriptor.parameter_string,
            file_name=file_descriptor.file_name)
        return glob.glob(file_pattern)



class FilepathGenerator:
    def __init__(self, instrument:str, base_path:str="data/processed", mode:str="decay_only", extension:str="wav" ):
        self.instrument = instrument
        self.base_path = base_path
        self.mode = mode
        self.extension = extension

    def generate_file_path(self, fret_no:str, exciter_type:str):
        return ('{base_path}/{mode}/{instrument}_{fret_no}_{exciter_type}.{extension}'
                .format(base_path=self.base_path,mode=self.mode,instrument=self.instrument,
                        fret_no=fret_no, exciter_type=exciter_type, extension=self.extension))


class TrainingDatasetCache:
    def __init__(self, path: str):
        self.path = path

    def save_dataset(self, dataset: NeuralNetworkDataset):
        cache_path = Path(self.path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inputs": dataset.inputs,
                "targets": dataset.targets,
            },
            self.path
        )

    def load_dataset(self) -> NeuralNetworkDataset | None:
        cache_path = Path(self.path)
        if not cache_path.exists():
            return None
        data = torch.load(self.path, map_location="cpu")
        return NeuralNetworkDataset(
            inputs=data["inputs"],
            targets=data["targets"],
        )


def make_lookahead_signal(x: np.ndarray, lookahead: int):
    pad = np.zeros((lookahead, *x.shape[1:]), dtype=x.dtype)
    return np.concatenate([x[lookahead:], pad], axis=0)


class TrainingDataGenerator:
    def __init__(self, training_parameters: TrainingParameters, training_file_descriptor: TrainingFileDescriptor,
                 delay, controls, training_dataset_cache: TrainingDatasetCache,
                 base_frequency: float):
        self.training_parameters = training_parameters
        self.training_file_descriptor = training_file_descriptor
        self.delay = delay
        self.controls = controls
        self.training_dataset_cache = training_dataset_cache
        self.base_frequency = base_frequency

    def get_or_create_dataset(self) -> NeuralNetworkDataset:
        dataset = self.training_dataset_cache.load_dataset()
        if dataset:
            print("Loading dataset from cache")
            return dataset

        print("Generating dataset")
        dataset = self.generate_training_dataset()  # teuer
        self.training_dataset_cache.save_dataset(dataset)
        return dataset
    def generate_training_dataset(self):


        file_finder = TrainingFileFinder()
        file_paths = file_finder.get_filepaths(self.training_file_descriptor)
        max_frames = int (self.training_parameters.max_training_data_frames / len(file_paths))
        accumulated_training_data = None
        for file_path in file_paths:
            training_data = self.generate_training_dataset_from_filepath(filepath=file_path, max_frames=max_frames)
            if not accumulated_training_data:
                accumulated_training_data = training_data
            else:
                accumulated_training_data.add(training_data)

        return accumulated_training_data

    def generate_training_dataloader(self):

        dataset = self.get_or_create_dataset()
        return prepare_dataloader(dataset, batch_size=self.training_parameters.batch_size)

    def generate_training_dataset_from_filepath(self, filepath: str, max_frames: int) -> NeuralNetworkDataset:
        input_feature_extractors = []
        input_feature_extractors.append(AudioMonoSplitFeatureExtractor(audio_processor=self.delay))
        input_feature_extractors.append(ConstantControlValueFeatureProvider([0.0]))
        period_delay_in_samples = window_size = 550
        sample_rate = 44100
        decay_meter = DecayMeter(window_size=window_size, sample_rate=int(sample_rate),time_in_samples=period_delay_in_samples)
        input_feature_extractors.append(AudioDecayMeterFeatureExtractor(decay_meter=decay_meter))
        output_feature_extractors = []
        output_feature_extractors.append(SimpleAudioFeatureExtractor())

        audio_training_data_generator = AudioTrainingDataGenerator(base_frequency=self.base_frequency,
                                                                   input_feature_extractors=input_feature_extractors,
                                                                   output_feature_extractors=output_feature_extractors)
        dataset = audio_training_data_generator.training_data_from_audio_file(filepath)

        random_dataset_reducer = RandomTrainingDatasetReducer(ReproRNG(100))
        dataset = random_dataset_reducer.reduce_dataset(dataset, max_frames)

        burst_manipulator = RandomAudioBurstDatasetManipulator([[0,21], [21,42], [42,63]])
        dataset = burst_manipulator.manipulate_dataset(dataset)

        negative_version_manipulator = SymmetricVersionDatasetManipulator(input_indices=[[0, len(self.delay.delays)]], output_indices=[[0,1]])
        dataset = negative_version_manipulator.manipulate_dataset(dataset)



        return dataset

    def xxxgenerate_training_dataset_from_filepath(self,  filepath: str, max_frames):
        # WAV-Datei laden
        signal, samplerate = sf.read(filepath, dtype='float32')
        if signal.ndim == 2:
            signal = signal[:, 0]



        # for initialization, we need to feed at least so many samples into the multi-tap delay that the longest
        # of delays is completely full and outputs the first sample
        period_delay_in_samples = 550 # TODO: proper length of delay
        max_delay_samples = period_delay_in_samples
        num_lookahead_samples = period_delay_in_samples
        window_size = period_delay_in_samples
        init_samples = signal[:(max_delay_samples + num_lookahead_samples)]
        decay_inertia = pow(0.5,1/period_delay_in_samples)
        self.delay.prepare()
        self.delay.process_mono_split(init_samples)  # output can be ignored
        remaining_signal = signal[(max_delay_samples + num_lookahead_samples):]
        average_decay = None
        decay_meter = None

        if self.training_parameters.use_energy_and_decay:
            # window_size = int(self.training_parameters.energy_window_size_in_s * samplerate)
            time_in_samples = 5*int(samplerate) # 5 seconds for average decay measuring
            average_decay_meter = DecayMeter(window_size=window_size, sample_rate=int(samplerate),time_in_samples=time_in_samples)
            decay_output = average_decay_meter.process_mono(signal[:(time_in_samples + window_size + 1)])
            # average decay per period in dB
            average_decay = decay_output[-1]*period_delay_in_samples/time_in_samples
            decay_meter = DecayMeter(window_size=window_size, sample_rate=int(samplerate),time_in_samples=period_delay_in_samples)
            num_lookahead_samples = period_delay_in_samples
            decay_meter.prepare()
            decay_meter.process_mono(init_samples)

        input_list = []
        target_list = []
        # iterate per sample over rest of audio file:
        # Feed the delay with the signal, the delay's output is the input of the neural network + control signal
        count = 0
        controls_input = self.controls.get_control_input_data()
        controls_row = controls_input.reshape(1, -1)

        skip_probability = 1 - min(1, max_frames/len(remaining_signal))
        # reproducible random number generator to get reproducible results while randomly reducing training data
        rng = ReproRNG(100)

        batch_size = 1
        main_gen = frame_batch_generator(remaining_signal, batch_size)
        decay_parameter = 0
        for frame in main_gen:

            delay_output = self.delay.process_mono_split(frame)

            if self.training_parameters.use_energy_and_decay:
                decay_output = decay_meter.process_mono(frame)
                # Map roughly +-0.04 scale to -1...1, with average at 0.
                decay_scale = 0.04
                new_decay_parameter = float((decay_output[0] - average_decay) / decay_scale)
                decay_parameter = decay_inertia*decay_parameter + (1-decay_inertia)*new_decay_parameter
                # apply minmax clipping only on actual training parameter, not before applying inertia.
                decay_row = np.array([[min(max(decay_parameter,-1),1)]])
                decay_target = np.array([[new_decay_parameter]])
            else:
                decay_row = None
                decay_target = None
            if skip_probability and rng.bernoulli(skip_probability):
                # Reduce Training data so that we can also use the tail of the file without slowing down training too much
                continue

            delay_row = delay_output.T  # Transpose because trainingdata expects rows instead of columns (Trainingdata: 1 row = 1 Trainingset, 1 colum = 1 Feature)
            concatenate_rows = [delay_row, controls_row]
            if decay_row:
                concatenate_rows.append(decay_row)

            inputs_concatenated = np.concatenate(concatenate_rows, axis=1)
            target_list.append(frame)
            input_list.append(inputs_concatenated)

            # force symmetric behaviour by adding symmetric training sample
            negative_delay_row = delay_row * -1
            negative_frame = frame * -1
            concatenate_rows = [negative_delay_row, controls_row]
            if decay_row:
                concatenate_rows.append(decay_row)
            negative_inputs_concatenated = np.concatenate(concatenate_rows, axis=1)
            target_list.append(negative_frame)
            input_list.append(negative_inputs_concatenated)

            count += 1
            if count > max_frames:
                break

        targets = torch.tensor(np.vstack(target_list), dtype=torch.float32)  # Shape: (N, 1)
        inputs = torch.tensor(np.vstack(input_list), dtype=torch.float32)
        return NeuralNetworkDataset(inputs, targets)




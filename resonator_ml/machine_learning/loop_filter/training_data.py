from typing import Callable

import torch
import glob
from pathlib import Path

from dataclasses import dataclass, field
import json
from torch.utils.data import DataLoader
from hashlib import sha256

from app.config.app import Config
from resonator_ml.audio.metering import DecayMeter
from resonator_ml.machine_learning.loop_filter.neural_network import DelayPattern
from resonator_ml.machine_learning.training.data import NeuralNetworkDataset, AudioTrainingDataGenerator, \
    AudioMonoSplitFeatureExtractor, ConstantControlValueFeatureProvider, AudioDecayMeterFeatureExtractor, \
    SimpleAudioFeatureExtractor, RandomTrainingDatasetReducer, RandomAudioBurstDatasetManipulator, \
    SymmetricVersionDatasetManipulator, TrainingDatasetSequencer
from resonator_ml.machine_learning.training.parameters import TrainingParameters
from resonator_ml.utility.random import ReproRNG

TRAINING_DATA_BASE_PATH = "data/processed"
TRAINING_DATA_SUB_PATH_DECAY = "decay_only"


def prepare_dataloader(dataset, batch_size=512):

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

@dataclass()
class CacheKeyRequest:
    delay_patterns: list[DelayPattern] = field(default_factory=lambda : [])
    max_training_data_frames: int = 0
    use_decay_feature: bool = False
    instrument_name: str = ""
    sub_cache_id: str = "" # can be filename or {filename}_processed or "all" or just blank ""

class TrainingDataCacheKeyProvider:
    def __init__(self, config: Config, cache_key_generator: 'TrainingDataCacheKeyGenerator'):
        self.config = config
        self.cache_key_generator = cache_key_generator

    def provide_cache_key_for_sub_id(self, sub_id: str = "") -> str:
        request = CacheKeyRequest(
            delay_patterns=self.config.neural_network_parameters.delay_patterns,
            max_training_data_frames=self.config.training_parameters.max_training_data_frames,
            use_decay_feature=self.config.training_parameters.use_energy_and_decay,
            instrument_name=self.config.instrument_name,
            sub_cache_id=sub_id
        )
        return self.cache_key_generator.training_data_cache_key(request)
class TrainingDataCacheKeyGenerator:
    def training_data_cache_key(self, request: CacheKeyRequest) -> str:
        serialized_patters = ""
        for pattern in request.delay_patterns:
            serialized_patters = "+" + serialized_patters + str(pattern.n_before) + "_" + str(pattern.n_after) + "_" + str(pattern.t_factor)+ "-"
        cache_key_object = [
            request.max_training_data_frames,
            request.use_decay_feature,
            serialized_patters,
            request.instrument_name,
            request.sub_cache_id
        ]
        return '{instrument}_{sub_id}_{hash}.tdata'.format(
                    instrument=request.instrument_name, sub_id= request.sub_cache_id[:7], hash=sha256(json.dumps(cache_key_object, sort_keys=True).encode("utf-8")).hexdigest())


class TrainingDatasetCache:
    def __init__(self, path: str, cache_key_provider: TrainingDataCacheKeyProvider):
        self.path = path
        self.cache_key_provider = cache_key_provider

    def save_dataset(self, dataset: NeuralNetworkDataset, sub_id: str = ""):
        cache_key = self.cache_key_provider.provide_cache_key_for_sub_id(sub_id)
        cache_path = Path(self.path) / cache_key
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inputs": dataset.inputs,
                "targets": dataset.targets,
            },
            cache_path
        )

    def load_dataset(self, sub_id: str = "") -> NeuralNetworkDataset | None:
        cache_key = self.cache_key_provider.provide_cache_key_for_sub_id(sub_id)
        cache_path = Path(self.path) / cache_key
        if not cache_path.exists():
            return None
        data = torch.load(cache_path, map_location="cpu")
        return NeuralNetworkDataset(
            inputs=data["inputs"],
            targets=data["targets"],
        )



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

    def get_or_create_dataset(self, factory: Callable[[], NeuralNetworkDataset], sub_id: str = "") -> NeuralNetworkDataset:
        dataset = self.training_dataset_cache.load_dataset(sub_id)
        if dataset:
            print("Loading dataset from cache")
            return dataset

        print("Generating dataset")
        dataset = factory() # teuer
        self.training_dataset_cache.save_dataset(dataset, sub_id)
        return dataset

    def generate_training_dataset(self):


        file_finder = TrainingFileFinder()
        file_paths = file_finder.get_filepaths(self.training_file_descriptor)
        max_frames = int (self.training_parameters.max_training_data_frames / len(file_paths))
        accumulated_training_data = None
        for file_path in file_paths:
            training_data = self.get_or_create_dataset(
                lambda: self.generate_training_dataset_from_filepath(filepath=file_path, max_frames=max_frames),
                sub_id="full_" + file_path)
            if not accumulated_training_data:
                accumulated_training_data = training_data
            else:
                accumulated_training_data.add(training_data)

        return accumulated_training_data

    def generate_training_dataloader(self):

        dataset = self.get_or_create_dataset(lambda : self.generate_training_dataset(), "all")
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

        dataset = self.get_or_create_dataset(lambda : audio_training_data_generator.training_data_from_audio_file(filepath),
                                             sub_id="partial_" + filepath)

        # burst_manipulator = RandomAudioBurstDatasetManipulator([[0,21], [21,42], [42,63], [42,84]])
        # dataset = burst_manipulator.manipulate_dataset(dataset)

        negative_version_manipulator = SymmetricVersionDatasetManipulator(input_indices=[[0, len(self.delay.delay_times)]],
                                                                          output_indices=[[0, 1]])
        negative_dataset = negative_version_manipulator.manipulate_dataset(dataset)
        sequencer = TrainingDatasetSequencer()
        sequenced_dataset = sequencer.sequence_dataset(dataset)
        negative_sequenced_dataset = sequencer.sequence_dataset(negative_dataset)

        sequenced_dataset.add(negative_sequenced_dataset)

        random_dataset_reducer = RandomTrainingDatasetReducer(ReproRNG(100))
        dataset = random_dataset_reducer.reduce_dataset(sequenced_dataset, max_frames)

        return dataset




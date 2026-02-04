from app.config.app import Config
from pathlib import Path
from typing import Generator, Any
import json
from resonator_ml.ports.file_storage import FileStorage, DictStorage
from hashlib import sha256

class LocalFileSystemStorage(FileStorage):
    def __init__(self, config: Config):
        self.config = config

    def model_file_path(self) -> Path:
        path = self._output_folder_path() / 'model.pt'
        return path

    def history_dirs(self) -> list[Path]:
        base_path = self._output_folder_base_path()
        numeric_dirs = sorted(
            (
                p for p in base_path.iterdir()
                if p.is_dir() and p.name.isdigit()
            ),
            key=lambda p: int(p.name)
        )

        return numeric_dirs

    def _output_folder_path(self) -> Path|None:


        base_path = self._output_folder_base_path()
        current_version = self._current_path_version()
        if not current_version:
            return None

        path = base_path / str(current_version)

        return path

    def _current_path_version(self) -> int|None:
        history_dirs = self.history_dirs()
        max_number = max(
            (int(p.name) for p in history_dirs),
            default=None
        )
        return max_number

    def _output_folder_base_path(self) -> Path:
        path = Path('.')
        path = path / self.config.results_path / self.config.resonator_results_sub_path / self.config.instrument_name
        return path

    def make_new_version_output_dir(self) -> Path:
        base_path = self._output_folder_base_path()
        current_version = self._current_path_version()
        if not current_version:
            current_version = 1
        else:
            current_version = current_version + 1
        path = base_path / str(current_version)
        path.mkdir()
        return path

    def sound_output_path(self) -> Path:
        path = self._output_folder_path() / 'output.wav'
        return path

    def parameters_output_path(self) -> Path:
        path = self._output_folder_path() / 'params.json'
        return path

    def training_data_cache_path(self) -> Path:
        path = Path('.')
        # TODO: Cache key somewhere else
        serialized_patters = ""
        for pattern in self.config.neural_network_parameters.delay_patterns:
            serialized_patters = "+" + serialized_patters + str(pattern.n_before) + "_" + str(pattern.n_after) + "_" + str(pattern.t_factor)+ "-"
        cache_key_object = [
            self.config.training_parameters.max_training_data_frames,
            self.config.neural_network_parameters.use_decay_feature,
            serialized_patters,
            self.config.instrument_name
        ]
        path = (path / self.config.cache_path / self.config.loop_filer_training_data_cache_sub_path /
                '{instrument}_{hash}.tdata'.format(
                    instrument=self.config.instrument_name, hash=sha256(json.dumps(cache_key_object, sort_keys=True).encode("utf-8")).hexdigest()))
        return path

    def training_file_paths(self, parameter_string: str) -> Generator[Path, None, None]:
        folder = '{base_path}/{model_name}/{parameter_string}'.format(
            base_path=self.config.resonator_training_path,
            model_name=self.config.instrument_name, parameter_string=parameter_string)
        path = Path(folder)
        return path.glob("*.war")



class DictJsonFileLogger(DictStorage):
    def __init__(self, path: Path):
        self.path = path
    def save_dict(self, params: dict[str, Any]):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def load_dict(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)
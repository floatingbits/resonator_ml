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
        if self.config.model_file_path:
            return Path(self.config.model_file_path)
        output_path = self._output_folder_path()
        if output_path:
            path = output_path / 'model.pt'
            return path
        else:
            raise FileNotFoundError('No output dir yet, so no model path')

    def src_model_file_path(self) -> Path:
        if self.config.src_model_file_path:
            return Path(self.config.src_model_file_path)
        return self.model_file_path()

    def history_dirs(self) -> list[Path]:
        base_path = self.output_folder_base_path()
        return self._numeric_dirs_in_path(base_path)

    def _numeric_dirs_in_path(self, path:Path) -> list[Path]:
        numeric_dirs = sorted(
            (
                p for p in path.iterdir()
                if p.is_dir() and p.name.isdigit()
            ),
            key=lambda p: int(p.name)
        )

        return numeric_dirs

    def _output_folder_path(self) -> Path|None:


        base_path = self.output_folder_base_path()
        current_version = self._current_path_version()
        if not current_version:
            return None

        path = base_path / str(current_version)

        return path

    def _current_path_version(self) -> int|None:
        history_dirs = self.history_dirs()
        return self._current_path_version_for_dirs(history_dirs)

    def _current_path_version_for_dirs(self, dirs: list[Path]) -> int|None:
        max_number = max(
            (int(p.name) for p in dirs),
            default=None
        )
        return max_number

    def _instrument_base_path(self) -> Path:
        path = Path('.')
        path = path / self.config.results_path / self.config.resonator_results_sub_path
        if self.config.experiment_name:
            path = path / "experiments" / self.config.experiment_name
        path = path / self.config.instrument_name
        return path



    def output_folder_base_path(self) -> Path:
        path = self._instrument_base_path()
        path.mkdir(parents=True, exist_ok=True)
        if self.config.experiment_name:
            experiment_run_dirs = self._numeric_dirs_in_path(path)
            experiment_run_id = self._current_path_version_for_dirs(experiment_run_dirs)
            experiment_run_id = 1 if experiment_run_id is None else experiment_run_id
            path = path / str(experiment_run_id)
        return path

    def _make_new_version_dir(self, base_path, current_version) -> Path:
        if not current_version:
            current_version = 1
        else:
            current_version = current_version + 1
        path = base_path / str(current_version)
        path.mkdir()
        return path

    def make_new_experiment_run_dir(self) -> Path:
        base_path = self._instrument_base_path()
        current_version = self._current_path_version_for_dirs(self._numeric_dirs_in_path(base_path))
        return self._make_new_version_dir(base_path,current_version)

    def make_new_version_output_dir(self) -> Path:
        base_path = self.output_folder_base_path()
        current_version = self._current_path_version()
        return self._make_new_version_dir(base_path,current_version)

    def sound_output_path(self) -> Path:
        path = self._output_folder_path() / 'output.wav'
        return path

    def parameters_output_path(self) -> Path:
        path = self._output_folder_path() / 'params.json'
        return path

    def results_output_path(self) -> Path:
        path = self._output_folder_path() / 'results.json'
        return path

    def training_data_cache_path(self) -> Path:
        path = Path('.')
        path = (path / self.config.cache_path / self.config.loop_filer_training_data_cache_sub_path)
        return path

    def training_file_paths(self, parameter_string: str) -> Generator[Path, None, None]:
        folder = '{base_path}/{model_name}/{parameter_string}'.format(
            base_path=self.config.resonator_training_path,
            model_name=self.config.instrument_name, parameter_string=parameter_string)
        path = Path(folder)
        return path.glob("*.wav")



class DictJsonFileLogger(DictStorage):
    def __init__(self, path: Path):
        self.path = path
    def save_dict(self, params: dict[str, Any]):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def load_dict(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)
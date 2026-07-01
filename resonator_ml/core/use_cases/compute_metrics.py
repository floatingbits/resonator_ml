from resonator_ml.audio.io import load_wav_mono
from resonator_ml.audio.metrics import Metric
from resonator_ml.audio.util import normalize
from pathlib import Path
import errno

from resonator_ml.ports.file_storage import DictStorage


class ComputeMetrics:
    def __init__(self, reference_wav_path: str, test_wav_path: str, metrics:dict[str, Metric], result_storage:DictStorage):
        self.reference_wav_path = reference_wav_path
        self.test_wav_path = test_wav_path
        self.metrics = metrics
        self.result_storage = result_storage

    def execute(self, config_id:int=0, config_run_id:int=0, config=None):
        if config is None:
            config = {}
        if not Path(self.reference_wav_path).exists():
            raise FileNotFoundError(errno.ENOENT, "Reference file not found", self.reference_wav_path)
        if not Path(self.test_wav_path).exists():
            raise FileNotFoundError(errno.ENOENT, "Test file not found. Did you generate it?", self.test_wav_path)
        sr1, ref = load_wav_mono(self.reference_wav_path)
        sr2, test = load_wav_mono(self.test_wav_path)
        assert sr1 == sr2, "Samplerates must be identical"
        metric_results = {}

        ref, test = self._prepare_wavs_for_metrics(ref,test)
        for metric_title in self.metrics:
            metric_results[metric_title] = float(self.metrics[metric_title].compute(ref,test,sr1))

        results_dict = {
            "results": metric_results,
            "config_id": config_id,
            "config_run_id": config_run_id,
        }
        if config:
            results_dict['config'] = config
        self.result_storage.save_dict(results_dict)
        print(metric_results)
        return results_dict

    def _prepare_wavs_for_metrics(self,ref,test):
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]

        return normalize(ref), normalize(test)
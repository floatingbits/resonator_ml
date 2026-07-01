from app.bootstrap import build_train_loop_network_use_case, build_generate_sound_file_use_case, \
    build_compute_metrics_use_case
from app.config.app import Config
from app.orchestration.experiment.definition import ExperimentDefinition
from resonator_ml.ports.file_storage import FileStorage
import copy

class AutoTrainer:
    def __init__(self, config: Config, file_storage: FileStorage):
        self.file_storage = file_storage
        self.config = config
    def run(self, config_extractor, determine_break_condition, modify_config):
        self.file_storage.make_new_experiment_run_dir()
        break_condition = False
        config = copy.deepcopy(self.config)
        best_config = copy.deepcopy(self.config)
        c = 0
        r = 0
        if config.reuse_last_model_file or config.src_model_file_path is not None:
            config = modify_config(config)
        last_combined_metric = 10.5 # known for src model file. Put in config...
        while not break_condition:
            build_train_loop_network_use_case(config).execute()
            build_generate_sound_file_use_case(config).execute()
            result_metrics = build_compute_metrics_use_case(config).execute(c,r, config_extractor(config))
            break_condition = determine_break_condition(c, result_metrics)
            # df['sc_scaled'] = np.minimum(15 * (df[evaluation_data.result_fields[1]] - 0.3), 20)
            # df['lsfft_scaled'] = np.minimum(5 * (df[evaluation_data.result_fields[0]] - 4.5), 10)
            # df['mel_scaled'] = np.minimum(0.8 * (df[evaluation_data.result_fields[2]] - 14), 10)
            combined_metric = (
                    min(15 * (result_metrics['results']['spectral_convergence'] - 0.3),20) +
                    min(5 * (result_metrics['results']['log_stft'] - 4.5),10) +
                    min(0.8 * (result_metrics['results']['mel_distance'] - 14), 10)
            )
            r += 1
            if not break_condition:
                has_improved = last_combined_metric is None or combined_metric < last_combined_metric
                if has_improved:
                    best_config = config
                config = modify_config(copy.deepcopy(best_config), result_metrics, has_improved)
                # Force re-using last model -> idea of autotrainer loop with metrics evaluation
                config.reuse_last_model_file = True

                if not has_improved:
                    # use backup path for next round
                    config.src_model_file_path = self.file_storage.model_file_path().parent / (self.file_storage.model_file_path().name + '.bak')
                else:
                    # only count, re-use the currently built model and update metric is we had an improvement.
                    c += 1
                    r = 0
                    config.src_model_file_path = None
                    last_combined_metric = combined_metric




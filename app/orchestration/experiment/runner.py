from app.bootstrap import build_train_loop_network_use_case, build_generate_sound_file_use_case, \
    build_compute_metrics_use_case
from app.orchestration.experiment.definition import ExperimentDefinition
from resonator_ml.ports.file_storage import FileStorage


class ExperimentRunner:
    def __init__(self, file_storage: FileStorage):
        self.file_storage = file_storage
    def run(self, experiment: ExperimentDefinition):
        self.file_storage.make_new_experiment_run_dir()
        for c,config in enumerate(experiment.configs):
            #results = []

            for i in range(experiment.runs_per_config):
                build_train_loop_network_use_case(config).execute()
                build_generate_sound_file_use_case(config).execute()
                build_compute_metrics_use_case(config).execute(c,i, experiment.config_extractor(config))


            #    results.append(metric)

           # self._aggregate(config, results)

from app.bootstrap import build_train_loop_network_use_case, build_generate_sound_file_use_case, \
    build_compute_metrics_use_case


class DefaultAudioOrchestrator:
    def run(self, config):
        build_train_loop_network_use_case(config).execute()
        build_generate_sound_file_use_case(config).execute()
        build_compute_metrics_use_case(config).execute()

from app.bootstrap import build_show_result_metrics_use_case
from app.config.app import Config as AppConfig
from app.orchestration.simple import DefaultAudioOrchestrator


def run(config: AppConfig):
    use_case = build_show_result_metrics_use_case(config)
    use_case.execute()


from app.bootstrap import build_compare_models_use_case
from app.config.app import Config as AppConfig


def run(config: AppConfig):
    use_case = build_compare_models_use_case(config)
    use_case.execute()

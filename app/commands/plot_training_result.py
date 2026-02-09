from app.bootstrap import build_plot_training_result_use_case
from app.config.app import Config as AppConfig


def run(config: AppConfig):
    training_result_use_case = build_plot_training_result_use_case(config)
    training_result_use_case.execute()
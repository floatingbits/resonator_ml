from app.bootstrap import build_plot_training_data_use_case, build_plot_weights_use_case
from app.config.app import Config as AppConfig

def run(config: AppConfig):
    use_case = build_plot_weights_use_case(config)
    use_case.execute()

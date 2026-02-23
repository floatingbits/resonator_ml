from app.bootstrap import build_plot_training_data_use_case, build_plot_spectrum_comparison
from app.config.app import Config as AppConfig

def run(config: AppConfig):
    use_case = build_plot_spectrum_comparison(config)
    use_case.execute()

from resonator_ml.machine_learning.view.training import TimeSeriesPlotter
import numpy as np
from app.bootstrap import build_plot_training_result_use_case

if __name__ == "__main__":
    training_result_use_case = build_plot_training_result_use_case()
    training_result_use_case.execute()

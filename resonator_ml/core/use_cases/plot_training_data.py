from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator
from resonator_ml.machine_learning.view.audio import BatchFeatureViewer
from resonator_ml.machine_learning.view.training import TimeSeriesPlotter
import random

import matplotlib.pyplot as plt

class PlotTrainingData:
    def __init__(self, training_data_generator: TrainingDataGenerator):
        self.training_data_generator = training_data_generator

    def execute(self):

        dataloader = self.training_data_generator.generate_training_dataloader()

        for inputs, target, ids in dataloader:
            viewer = BatchFeatureViewer(inputs, target)
            break

        plt.show()
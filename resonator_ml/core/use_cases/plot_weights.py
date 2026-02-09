from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator
from resonator_ml.machine_learning.view.audio import BatchFeatureViewer
from resonator_ml.machine_learning.view.training import TimeSeriesPlotter
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class PlotWeights:
    def __init__(self, model):
        self.model = model

    def execute(self):


        # Custom colormap: pink -> black -> green
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pink_black_green",
            [
                (0.0, "#ff77aa"),  # -1  → helles Pink
                (0.5, "#000000"),  # 0  → schwarz
                (1.0, "#77ff77"),  # +1  → helles Grün
            ]
        )

        # Normierung: -1 .. 0 .. +1 symmetrisch
        norm = mcolors.TwoSlopeNorm(vmin=-.2, vcenter=0.0, vmax=.2)

        W1 = self.model.common_net[0].weight.detach().cpu()
        W2 = self.model.audio_head.weight.detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

        im1 = axes[0].imshow(W1, cmap=cmap, norm=norm, aspect="auto")
        axes[0].set_title("W1: Input → Hidden")
        axes[0].set_xlabel("Input-Dimension")
        axes[0].set_ylabel("Hidden-Neuron")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(W2, cmap=cmap, norm=norm, aspect="auto")
        axes[1].set_title("W2: Hidden → Output")
        axes[1].set_xlabel("Hidden-Neuron")
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()